"""
==============================================================================
Country ETF Rotation Strategy v1.0
==============================================================================
Based on Market Regime (First Method) with Optimal Parameters

Strategy Logic:
- 각 국가의 경기 국면(Regime)을 기반으로 투자 비중 결정
- 팽창/회복 국면에만 투자, 침체/둔화 국면은 제외
- Score 비례로 비중 배분 (팽창 > 회복 > 둔화 > 침체)

Author: GD
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

from regime_provider import RegimeProvider, COUNTRY_MAP


# =============================================================================
# 1. 데이터 로딩
# =============================================================================
provider = RegimeProvider(use_cache=True)
countries = provider.loaded_countries  # ['USA', 'Korea', 'China', ...]

# 티커 매핑: 국가명 -> ETF 티커
ticker_map = {c: COUNTRY_MAP[c]['ticker'] for c in countries}
# 예: {'USA': 'SPY', 'Korea': 'EWY', 'China': 'MCHI', ...}

# 투자 유니버스: 9개 국가
Univ = ['USA', 'Korea', 'China', 'Japan', 'Germany', 'France', 'UK', 'India', 'Brazil']
ETF_Univ = [ticker_map[x] for x in Univ]


# =============================================================================
# 2. 파라미터 설정 (최적화된 값)
# =============================================================================
"""
파라미터 설명:
- start_date: 백테스트 시작일
- min_score: 투자 최소 Score 임계값 (이 값 초과인 국가만 투자)
  - 1.0 = 둔화(1) 초과 → 회복(2), 팽창(3)만 투자
  - 0.5 = 침체(0) 초과 → 둔화 이상 투자
- max_weight: 단일 국가 최대 비중 (1.0 = 제한 없음)
- score_power: Score 거듭제곱 (1.0 = 선형, 2.0 = 제곱 → 팽창 집중)
- cost: 거래비용 (0.001 = 10bps = 0.10%)
"""

start_date = '2010-01-01'
min_score = 1.0      # Score > 1.0 인 국가만 투자 (회복/팽창)
max_weight = 1.0     # 단일 국가 최대 비중 제한 없음
score_power = 1.0    # Score 선형 비례
cost = 0.001         # 거래비용 10bps


# =============================================================================
# 3. Regime 데이터 로딩 함수
# =============================================================================
def get_regime_df(method: str) -> pd.DataFrame:
    """
    특정 method의 Regime DataFrame 생성
    
    Args:
        method: 'first', 'fresh', 'smart' 중 하나
        - first (exp1_regime): 지표 발표 즉시 반영 (가장 민감)
        - fresh (exp2_regime): 1개월 지연 반영
        - smart (exp3_regime): 지표 발표 주기 고려 (가장 보수적)
    
    Returns:
        DataFrame: index=trade_date, columns=국가명, values=Regime명
    """
    # method에 따른 컬럼명 매핑
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
    
    # 모든 국가 병합
    regime_df = pd.concat(all_dfs, axis=1).sort_index()
    
    # 결측값은 이전 값으로 채움 (forward fill)
    regime_df = regime_df.ffill()
    
    return regime_df


# =============================================================================
# 4. Score 매핑
# =============================================================================
"""
Score 체계:
- 팽창(3): 경기 확장 국면 → 가장 높은 비중
- 회복(2): 경기 회복 국면 → 중간 비중
- 둔화(1): 경기 둔화 시작 → 현재 전략에서는 제외 (min_score=1.0)
- 침체(0): 경기 침체 국면 → 제외
- Cash(-1): 현금 신호 → 제외
- Half(-2): 절반 투자 신호 → 제외
- Skipped(-3): 데이터 부족 → 제외
"""
score_map = {
    '팽창': 3, 
    '회복': 2, 
    '둔화': 1, 
    '침체': 0, 
    'Cash': -1, 
    'Half': -2, 
    'Skipped': -3
}


# =============================================================================
# 5. Weight 계산 함수 (핵심 로직)
# =============================================================================
def calc_weight(score_df: pd.DataFrame, univ: list, 
                max_w: float = 1.0, 
                min_score: float = 0, 
                score_power: float = 1.0) -> pd.DataFrame:
    """
    Score 기반 투자 비중 계산
    
    Args:
        score_df: 국가별 Score DataFrame (index=trade_date, columns=국가명)
        univ: 투자 대상 국가 리스트
        max_w: 단일 국가 최대 비중 (0~1)
        min_score: 투자 최소 Score (이 값 초과만 투자)
        score_power: Score 거듭제곱 (집중도 조절)
    
    Returns:
        DataFrame: 투자 비중 (index=trade_date, columns=국가명+CASH)
    
    Example:
        min_score=1.0 일 때:
        - USA(Score=2), Korea(Score=3), Japan(Score=1)
        - 유효: USA(2), Korea(3) → 합계=5
        - 비중: USA=2/5=40%, Korea=3/5=60%, Japan=0%
    """
    
    # Step 1: min_score 초과인 Score만 유효로 처리
    # 나머지는 NaN으로 마스킹
    valid = score_df[univ].where(score_df[univ] > min_score)
    
    # Step 2: Score Power 적용 (옵션)
    # score_power > 1이면 팽창(3)에 더 집중
    if score_power != 1.0:
        valid = valid ** score_power
    
    # Step 3: 각 날짜별로 Score 비례 비중 계산
    # 예: USA=2, Korea=3 → USA=2/(2+3)=40%, Korea=3/(2+3)=60%
    w = valid.div(valid.sum(axis=1), axis=0)
    
    # Step 4: 결측값 제거 (모든 국가가 NaN인 날짜 제외)
    w = w.dropna(how='all', axis=0)
    
    # Step 5: 최대 비중 제한 적용
    # max_w=0.5이면 한 국가 최대 50%
    w = w.clip(upper=max_w)
    
    # Step 6: NaN을 0으로 채움
    w = w.fillna(0)
    
    # Step 7: CASH 비중 계산
    # 국가 비중 합이 100% 미만이면 잔여분은 CASH
    w['CASH'] = (1 - w.sum(axis=1)).clip(lower=0)
    
    return w


# =============================================================================
# 6. 가격 데이터 로딩
# =============================================================================
gdu.data = provider._load_price_data()
gdu.data['CASH'] = 1  # CASH는 항상 가격=1 (수익률 0%)


def to_ticker(w_df: pd.DataFrame) -> pd.DataFrame:
    """국가명을 ETF 티커로 변환"""
    return w_df.rename(columns=lambda x: ticker_map.get(x, x))


# =============================================================================
# 7. 전략 실행
# =============================================================================
if __name__ == '__main__':
    
    print("="*80)
    print("COUNTRY ETF ROTATION STRATEGY v1.0")
    print("="*80)
    print(f"Period: {start_date} ~ present")
    print(f"Method: First (exp1_regime)")
    print(f"Parameters: min_score={min_score}, max_weight={max_weight}, power={score_power}")
    print(f"Cost: {cost*10000:.0f}bps")
    print("="*80)
    
    # 7-1. Regime 데이터 로딩 (First Method 사용)
    regime_first = get_regime_df('first')
    
    # 7-2. Score 변환
    score_first = regime_first.replace(score_map)
    
    # 7-3. Weight 계산
    w = calc_weight(score_first, Univ, max_weight, min_score, score_power)
    
    # 7-4. 티커로 변환
    w_ticker = to_ticker(w)
    
    # 7-5. 백테스트 실행
    ret = gdu.calc_return(w_ticker, cost=cost)
    
    # 7-6. 성과 평가
    def calc_perf(cum_ret, name=''):
        if cum_ret.empty or len(cum_ret) < 2:
            return None
        dr = cum_ret.pct_change().dropna()
        yrs = (cum_ret.index[-1] - cum_ret.index[0]).days / 365.25
        tot = cum_ret.iloc[-1] / cum_ret.iloc[0] - 1
        cagr = (1 + tot) ** (1/yrs) - 1 if yrs > 0 else 0
        vol = dr.std() * np.sqrt(252)
        sharpe = (cagr - 0.02) / vol if vol > 0 else 0
        rm = cum_ret.expanding().max()
        mdd = ((cum_ret - rm) / rm).min()
        return {'Name': name, 'CAGR': cagr, 'Vol': vol, 'Sharpe': sharpe, 'MDD': mdd}
    
    perf = calc_perf(ret, 'First_Strategy')
    
    print("\n" + "="*80)
    print("PERFORMANCE RESULT")
    print("="*80)
    print(f"CAGR:   {perf['CAGR']:.1%}")
    print(f"Vol:    {perf['Vol']:.1%}")
    print(f"Sharpe: {perf['Sharpe']:.2f}")
    print(f"MDD:    {perf['MDD']:.1%}")
    
    # 7-7. 현재 포지션
    print("\n" + "="*80)
    print("CURRENT POSITION")
    print("="*80)
    
    latest_w = w.iloc[-1]
    latest_regime = regime_first.iloc[-1]
    latest_score = score_first.iloc[-1]
    
    print(f"{'Country':<12} {'Regime':<8} {'Score':>6} {'Weight':>10} {'Ticker':<8}")
    print("-"*50)
    for c in Univ:
        if c in latest_w and latest_w[c] > 0:
            print(f"{c:<12} {latest_regime[c]:<8} {latest_score[c]:>6} {latest_w[c]:>9.1%} {ticker_map[c]:<8}")
    print(f"{'CASH':<12} {'':<8} {'':<6} {latest_w['CASH']:>9.1%}")
    print(f"\nLast Update: {w.index[-1].strftime('%Y-%m-%d')}")
    
    # 7-8. Weight 이력 (최근 5개)
    print("\n" + "="*80)
    print("WEIGHT HISTORY (Last 5 Rebalancing)")
    print("="*80)
    print((w[Univ + ['CASH']] * 100).round(1).tail(5).to_string())
    
    # 7-9. GUI 리포트 (선택)
    # gdu.report(ret.to_frame('Strategy'))
