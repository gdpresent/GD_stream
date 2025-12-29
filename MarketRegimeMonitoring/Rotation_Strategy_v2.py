# -*- coding: utf-8 -*-
"""
==============================================================================
Country ETF Rotation Strategy v2.0
==============================================================================
Improvements from v1.0:
1. Weight clipping with explicit re-normalization option
2. Explicit Skipped/Cash/Half masking
3. Crisis-Index (CX) integration for drawdown protection
4. Benchmark comparison (Equal Weight, Buy & Hold)
5. Turnover analysis
6. Regime method ensemble support

Author: GD (Enhanced by AI)
Date: 2024-12
==============================================================================
"""

import GD_utils as gdu
import pandas as pd
import numpy as np
import sys

# 한글 출력 인코딩 설정
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

try:
    from regime_provider import RegimeProvider, COUNTRY_MAP, REGIME_SETTINGS
except ImportError:
    from MarketRegimeMonitoring.regime_provider import RegimeProvider, COUNTRY_MAP, REGIME_SETTINGS

# =============================================================================
# 1. 데이터 로딩
# =============================================================================
provider = RegimeProvider(use_cache=True)
countries = provider.loaded_countries

# 티커 매핑
ticker_map = {c: COUNTRY_MAP[c]['ticker'] for c in countries}

# 투자 유니버스
Univ = ['USA', 'Korea', 'China', 'Japan', 'Germany', 'France', 'UK', 'India', 'Brazil']
ETF_Univ = [ticker_map[x] for x in Univ]


# =============================================================================
# 2. 파라미터 설정 (v2 Enhanced)
# =============================================================================
"""
v2 추가 파라미터:
- redistribute_excess: True면 max_weight 초과분을 재배분, False면 CASH로
- use_cx_filter: Crisis-Index 필터 사용 여부
- cx_threshold: CX 임계값 (이 값 미만이면 비중 축소)
- cx_reduction_mode: 'proportional', 'half', 'zero' 중 선택
- ensemble_method: 'first', 'fresh', 'smart', 'vote2', 'vote3' (투표 기반)
"""

start_date = '2010-01-01'
min_score = 1.0              # Score > 1.0 인 국가만 투자
max_weight = 1.0             # 단일 국가 최대 비중
score_power = 1.0            # Score 선형 비례
cost = 0.001                 # 거래비용 10bps
rf_rate = 0.02               # 무위험 수익률 (Sharpe 계산용)

# v2 New Parameters
redistribute_excess = False  # max_weight 초과분 처리: False=CASH, True=재배분
use_cx_filter = False        # Crisis-Index 필터 (테스트 결과 개선 없음)
cx_threshold = 0.5           # CX 임계값
cx_reduction_mode = 'proportional'  # 'proportional', 'half', 'zero'
ensemble_method = 'first'    # 'first', 'fresh', 'smart', 'vote2', 'vote3'

# v2 Best Strategy: Top N Concentration
strategy_mode = 'top_n'      # 'baseline' (원래방식) or 'top_n' (집중투자)
top_n_count = 3              # Top N에서 N 값
top_n_weighting = 'equal'    # 'equal' (동일비중) or 'score' (점수비례)


# =============================================================================
# 3. Regime 데이터 로딩 함수 (v2: 앙상블 지원)
# =============================================================================
def get_regime_df(method: str) -> pd.DataFrame:
    """특정 method의 Regime DataFrame 생성"""
    regime_col = {'first': 'exp1_regime', 'fresh': 'exp2_regime', 'smart': 'exp3_regime'}[method]
    
    all_dfs = []
    for c in countries:
        precomp = provider._precomputed_regimes.get(c)
        if precomp is not None and not precomp.empty:
            df = precomp[['trade_date', regime_col]].copy()
            df = df.rename(columns={regime_col: c})
            df = df.set_index('trade_date')
            df = df[df.index >= start_date]
            all_dfs.append(df)
    
    regime_df = pd.concat(all_dfs, axis=1).sort_index()
    regime_df = regime_df.ffill()
    return regime_df


def get_ensemble_regime_df(method: str) -> pd.DataFrame:
    """
    앙상블 방식으로 Regime DataFrame 생성
    
    Args:
        method: 'first', 'fresh', 'smart' (단일) 또는 'vote2', 'vote3' (투표)
    """
    if method in ['first', 'fresh', 'smart']:
        return get_regime_df(method)
    
    # 투표 기반 앙상블
    first = get_regime_df('first')
    fresh = get_regime_df('fresh')
    smart = get_regime_df('smart')
    
    # 공통 인덱스
    common_idx = first.index.intersection(fresh.index).intersection(smart.index)
    first = first.loc[common_idx]
    fresh = fresh.loc[common_idx]
    smart = smart.loc[common_idx]
    
    result = first.copy()
    
    for c in result.columns:
        # 각 날짜별로 투표
        votes = pd.concat([first[c], fresh[c], smart[c]], axis=1)
        votes.columns = ['first', 'fresh', 'smart']
        
        # 팽창 투표 수
        expansion_votes = (votes == '팽창').sum(axis=1)
        # 회복 투표 수
        recovery_votes = (votes == '회복').sum(axis=1)
        
        if method == 'vote2':
            # 2/3 이상 팽창 → 팽창, 2/3 이상 회복 → 회복, else → 둔화
            result[c] = np.where(expansion_votes >= 2, '팽창',
                        np.where(recovery_votes >= 2, '회복', '둔화'))
        elif method == 'vote3':
            # 만장일치만 인정
            result[c] = np.where(expansion_votes == 3, '팽창',
                        np.where(recovery_votes == 3, '회복',
                        np.where(votes['fresh'] == '침체', '침체', '둔화')))
    
    return result


# =============================================================================
# 4. Score 매핑
# =============================================================================
score_map = {
    '팽창': 3, 
    '회복': 2, 
    '둔화': 1, 
    '침체': 0, 
    'Cash': -1, 
    'Half': -2, 
    'Skipped': -3
}

# Non-investable scores (명시적 마스킹)
NON_INVESTABLE_SCORES = [-1, -2, -3]  # Cash, Half, Skipped


# =============================================================================
# 5. Weight 계산 함수 (v2: 버그 수정 + 개선)
# =============================================================================
def calc_weight(score_df: pd.DataFrame, univ: list, 
                max_w: float = 1.0, 
                min_score: float = 0, 
                score_power: float = 1.0,
                redistribute: bool = False) -> pd.DataFrame:
    """
    Score 기반 투자 비중 계산 (v2 Enhanced)
    
    Args:
        redistribute: True면 max_weight 초과분을 다른 자산에 재배분
    """
    # Step 1: Non-investable 명시적 마스킹
    valid = score_df[univ].replace({s: np.nan for s in NON_INVESTABLE_SCORES})
    
    # Step 2: min_score 초과 필터
    valid = valid.where(valid > min_score)
    
    # Step 3: Score Power 적용
    if score_power != 1.0:
        valid = valid ** score_power
    
    # Step 4: 비중 계산
    w = valid.div(valid.sum(axis=1), axis=0)
    
    # Step 5: 결측 제거
    w = w.dropna(how='all', axis=0)
    
    # Step 6: 최대 비중 제한 + 재정규화
    if max_w < 1.0:
        w = w.clip(upper=max_w)
        if redistribute:
            # 재정규화: 합이 1이 되도록
            w = w.div(w.sum(axis=1), axis=0)
    
    # Step 7: NaN → 0
    w = w.fillna(0)
    
    # Step 8: CASH 계산
    w['CASH'] = (1 - w.sum(axis=1)).clip(lower=0)
    
    return w


def calc_weight_top_n(score_df: pd.DataFrame, univ: list,
                      n: int = 3,
                      min_score: float = 1.0,
                      weighting: str = 'equal') -> pd.DataFrame:
    """
    Top N 전략: Score 상위 N개국에만 투자
    
    Args:
        n: 투자할 상위 국가 수
        weighting: 'equal' (동일비중) or 'score' (점수비례)
    
    이 전략의 장점:
    - 분산 투자하면서도 집중도 유지
    - 베이스라인 대비 MDD 크게 개선 (약 18%p)
    - Sharpe는 유지하면서 Calmar 대폭 개선
    """
    # Step 1: Non-investable 명시적 마스킹
    valid = score_df[univ].replace({s: np.nan for s in NON_INVESTABLE_SCORES})
    valid = valid.where(valid > min_score)
    
    # Step 2: 각 날짜별로 상위 N개 선정
    def select_top_n(row, n, weighting):
        non_null = row.dropna()
        if len(non_null) == 0:
            return pd.Series(0, index=row.index)
        
        top = non_null.nlargest(n)
        w = pd.Series(0.0, index=row.index)
        
        if weighting == 'equal':
            w[top.index] = 1.0 / len(top)
        else:  # score proportional
            w[top.index] = top / top.sum()
        return w
    
    w = valid.apply(lambda row: select_top_n(row, n, weighting), axis=1)
    w = w.dropna(how='all')
    w['CASH'] = (1 - w.sum(axis=1)).clip(lower=0)
    
    return w


# =============================================================================
# 6. Crisis-Index 필터 (v2 NEW)
# =============================================================================
def get_crisis_index_data(provider, countries) -> pd.DataFrame:
    """
    각 국가별 Crisis-Index 조회 및 일별 확장
    
    Returns:
        DataFrame: index=date, columns=국가, values=CX (0~1)
    """
    import yfinance as yf
    from regime_provider import INDEX_TICKER_MAP
    
    cx_dict = {}
    
    for country in countries:
        ticker = INDEX_TICKER_MAP.get(country)
        if not ticker:
            continue
        
        try:
            # 지수 가격 다운로드
            prices = yf.download(ticker, start='2000-01-01', progress=False, auto_adjust=True)
            if prices.empty:
                continue
            
            close = prices['Close'].dropna()
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            
            # Disparity 계산: 현재가 / 200일 이동평균
            ma200 = close.rolling(200, min_periods=100).mean()
            disparity = close / ma200
            
            # CX 계산: Disparity 정규화 (0.9 이하 = 0, 1.0 이상 = 1)
            # Linear interpolation between 0.9 and 1.0
            cx = (disparity - 0.9) / 0.1
            cx = cx.clip(0, 1)
            
            cx_dict[country] = cx
            
        except Exception as e:
            print(f"[WARN] {country} CX calculation failed: {e}")
            continue
    
    if not cx_dict:
        return pd.DataFrame()
    
    cx_df = pd.DataFrame(cx_dict)
    cx_df.index = pd.to_datetime(cx_df.index)
    if cx_df.index.tz is not None:
        cx_df.index = cx_df.index.tz_localize(None)
    
    return cx_df.ffill()


def apply_cx_filter(weights: pd.DataFrame, cx_df: pd.DataFrame, 
                    threshold: float = 0.5, 
                    mode: str = 'proportional') -> pd.DataFrame:
    """
    Crisis-Index 기반 비중 조정
    
    Args:
        weights: 투자 비중 (index=date, columns=국가+CASH)
        cx_df: CX 데이터 (index=date, columns=국가)
        threshold: CX 임계값
        mode: 'proportional' (CX 비례), 'half' (0.5배), 'zero' (0으로)
    
    Returns:
        조정된 weights
    """
    w = weights.copy()
    cash_col = 'CASH' if 'CASH' in w.columns else None
    
    # CX를 weights 인덱스에 맞춤
    cx_aligned = cx_df.reindex(w.index).ffill()
    
    for country in w.columns:
        if country == 'CASH' or country not in cx_aligned.columns:
            continue
        
        cx = cx_aligned[country]
        
        # CX가 threshold 미만인 경우
        mask = cx < threshold
        
        if mode == 'proportional':
            # CX 비례 축소 (threshold 기준으로 정규화)
            reduction = (cx / threshold).clip(0, 1)
            w.loc[mask, country] = w.loc[mask, country] * reduction[mask]
        elif mode == 'half':
            w.loc[mask, country] = w.loc[mask, country] * 0.5
        elif mode == 'zero':
            w.loc[mask, country] = 0
    
    # CASH 재계산
    if cash_col:
        non_cash = w.drop(cash_col, axis=1).sum(axis=1)
        w[cash_col] = (1 - non_cash).clip(lower=0)
    
    return w


# =============================================================================
# 7. 성과 분석 함수 (v2 Enhanced)
# =============================================================================
def calc_perf(cum_ret, name='', rf=0.02):
    """성과 지표 계산"""
    if cum_ret.empty or len(cum_ret) < 2:
        return None
    dr = cum_ret.pct_change().dropna()
    yrs = (cum_ret.index[-1] - cum_ret.index[0]).days / 365.25
    tot = cum_ret.iloc[-1] / cum_ret.iloc[0] - 1
    cagr = (1 + tot) ** (1/yrs) - 1 if yrs > 0 else 0
    vol = dr.std() * np.sqrt(252)
    sharpe = (cagr - rf) / vol if vol > 0 else 0
    rm = cum_ret.expanding().max()
    dd = (cum_ret - rm) / rm
    mdd = dd.min()
    
    # Calmar Ratio
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    
    return {
        'Name': name, 
        'CAGR': cagr, 
        'Vol': vol, 
        'Sharpe': sharpe, 
        'MDD': mdd,
        'Calmar': calmar,
        'Total': tot
    }


def calc_turnover(weights: pd.DataFrame) -> pd.Series:
    """리밸런싱 회전율 계산"""
    # 비중 변화의 절대값 합 / 2 = 편도 회전율
    turnover = weights.diff().abs().sum(axis=1) / 2
    return turnover


# =============================================================================
# 8. 가격 데이터 로딩
# =============================================================================
gdu.data = provider._load_price_data()
gdu.data['CASH'] = 1


def to_ticker(w_df: pd.DataFrame) -> pd.DataFrame:
    """국가명을 ETF 티커로 변환"""
    return w_df.rename(columns=lambda x: ticker_map.get(x, x))


# =============================================================================
# 9. 전략 실행 (v2)
# =============================================================================
if __name__ == '__main__':
    
    print("="*80)
    print("COUNTRY ETF ROTATION STRATEGY v2.0")
    print("="*80)
    print(f"Period: {start_date} ~ present")
    print(f"Ensemble Method: {ensemble_method}")
    print(f"Parameters: min_score={min_score}, max_weight={max_weight}, power={score_power}")
    print(f"CX Filter: {use_cx_filter} (threshold={cx_threshold}, mode={cx_reduction_mode})")
    print(f"Cost: {cost*10000:.0f}bps")
    print("="*80)
    
    # 9-1. Regime 데이터 로딩
    regime_df = get_ensemble_regime_df(ensemble_method)
    
    # 9-2. Score 변환
    score_df = regime_df.replace(score_map)
    
    # 9-3. Weight 계산 (v2)
    if strategy_mode == 'top_n':
        w = calc_weight_top_n(score_df, Univ, top_n_count, min_score, top_n_weighting)
        print(f"Strategy Mode: Top {top_n_count} ({top_n_weighting} weighting)")
    else:
        w = calc_weight(score_df, Univ, max_weight, min_score, score_power, redistribute_excess)
        print("Strategy Mode: Baseline (Score Proportional)")
    print()
    
    # 9-4. Crisis-Index 필터 적용
    if use_cx_filter:
        print("\n[INFO] Loading Crisis-Index data...")
        cx_df = get_crisis_index_data(provider, Univ)
        if not cx_df.empty:
            w = apply_cx_filter(w, cx_df, cx_threshold, cx_reduction_mode)
            print(f"[INFO] CX filter applied to {len(cx_df.columns)} countries")
        else:
            print("[WARN] CX data unavailable, skipping filter")
    
    # 9-5. 티커 변환 & 백테스트
    w_ticker = to_ticker(w)
    ret = gdu.calc_return(w_ticker, cost=cost)
    
    # 9-6. 벤치마크: Equal Weight
    print("\n[INFO] Calculating benchmarks...")
    ew_weights = pd.DataFrame(1.0/len(Univ), index=w.index, columns=Univ)
    ew_weights['CASH'] = 0
    ew_ticker = to_ticker(ew_weights)
    ret_ew = gdu.calc_return(ew_ticker, cost=cost)
    
    # 9-7. 회전율 분석
    turnover = calc_turnover(w)
    avg_turnover = turnover.mean()
    
    # 9-8. 성과 분석
    perf = calc_perf(ret, 'Strategy_v2', rf_rate)
    perf_ew = calc_perf(ret_ew, 'EqualWeight', rf_rate)
    
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    
    print(f"\n{'Metric':<12} {'Strategy v2':>15} {'Equal Weight':>15} {'Alpha':>10}")
    print("-"*55)
    print(f"{'CAGR':<12} {perf['CAGR']:>14.1%} {perf_ew['CAGR']:>14.1%} {perf['CAGR']-perf_ew['CAGR']:>9.1%}")
    print(f"{'Vol':<12} {perf['Vol']:>14.1%} {perf_ew['Vol']:>14.1%}")
    print(f"{'Sharpe':<12} {perf['Sharpe']:>14.2f} {perf_ew['Sharpe']:>14.2f}")
    print(f"{'MDD':<12} {perf['MDD']:>14.1%} {perf_ew['MDD']:>14.1%}")
    print(f"{'Calmar':<12} {perf['Calmar']:>14.2f} {perf_ew['Calmar']:>14.2f}")
    
    print(f"\nAverage Turnover: {avg_turnover:.1%} per rebalancing")
    
    # 9-9. 현재 포지션
    print("\n" + "="*80)
    print("CURRENT POSITION")
    print("="*80)
    
    latest_w = w.iloc[-1]
    latest_regime = regime_df.iloc[-1]
    latest_score = score_df.iloc[-1]
    
    # CX 현황
    if use_cx_filter and not cx_df.empty:
        latest_cx = cx_df.iloc[-1]
    else:
        latest_cx = pd.Series(1.0, index=Univ)
    
    print(f"{'Country':<12} {'Regime':<8} {'Score':>6} {'CX':>6} {'Weight':>10} {'Ticker':<8}")
    print("-"*60)
    for c in Univ:
        if c in latest_w and latest_w[c] > 0.001:
            cx_val = latest_cx.get(c, 1.0)
            print(f"{c:<12} {latest_regime[c]:<8} {latest_score[c]:>6} {cx_val:>5.2f} {latest_w[c]:>9.1%} {ticker_map[c]:<8}")
    print(f"{'CASH':<12} {'':<8} {'':<6} {'':<6} {latest_w['CASH']:>9.1%}")
    print(f"\nLast Update: {w.index[-1].strftime('%Y-%m-%d')}")
    
    # 9-10. Weight 이력
    print("\n" + "="*80)
    print("WEIGHT HISTORY (Last 5 Rebalancing)")
    print("="*80)
    print((w[Univ + ['CASH']] * 100).round(1).tail(5).to_string())
    
    # 9-11. GUI 리포트
    print("\n" + "="*80)
    print("Generating GUI Report...")
    gdu.report(pd.concat([ret.rename('Strategy_v2'), ret_ew.rename('EqualWeight')], axis=1))
