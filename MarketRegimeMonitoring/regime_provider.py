# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import warnings

# GD_utils는 로컬 환경에서만 사용 (Streamlit Cloud에서는 미사용)
try:
    import GD_utils as gdu
except ImportError:
    gdu = None  # Streamlit Cloud 환경
from typing import Dict, List, Optional, Union, Callable
from datetime import datetime
warnings.filterwarnings('ignore')

# 캐시 디렉토리 기본 경로
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')

# =============================================================================
# Configuration
# =============================================================================
COUNTRY_MAP = {
    'USA': {'fred': 'USALOLITOAASTSAM', 'ticker': 'SPY', 'name': '미국(USA)'},
    'Korea': {'fred': 'KORLOLITOAASTSAM', 'ticker': 'EWY', 'name': '한국(Korea)'},
    'China': {'fred': 'CHNLOLITOAASTSAM', 'ticker': 'MCHI', 'name': '중국(China)'},
    'Japan': {'fred': 'JPNLOLITOAASTSAM', 'ticker': 'EWJ', 'name': '일본(Japan)'},
    'Germany': {'fred': 'DEULOLITOAASTSAM', 'ticker': 'EWG', 'name': '독일(Germany)'},
    'France': {'fred': 'FRALOLITOAASTSAM', 'ticker': 'EWQ', 'name': '프랑스(France)'},
    'UK': {'fred': 'GBRLOLITOAASTSAM', 'ticker': 'EWU', 'name': '영국(UK)'},
    'India': {'fred': 'INDLOLITOAASTSAM', 'ticker': 'INDA', 'name': '인도(India)'},
    'Brazil': {'fred': 'BRALOLITOAASTSAM', 'ticker': 'EWZ', 'name': '브라질(Brazil)'},
    'G7': {'fred': 'G7LOLITOAASTSAM', 'ticker': 'ACWI', 'name': 'G7 Global'},
}

# 각국 대표지수 Yahoo Finance 티커 (Crisis-Index 용)
INDEX_TICKER_MAP = {
    'USA': '^GSPC',
    'Korea': '^KS11',
    'China': '000001.SS',
    'Japan': '^N225',
    'Germany': '^GDAXI',
    'France': '^FCHI',
    'UK': '^FTSE',
    'India': '^NSEI',
    'Brazil': '^BVSP',
    'G7': '^GSPC',  # G7은 S&P500 대용
}
REGIME_SETTINGS = {
    '팽창': {'weight': 2.0, 'score': 3, 'color': '#2ca02c', 'label': 'Expansion'},
    '회복': {'weight': 1.0, 'score': 2, 'color': '#ffce30', 'label': 'Recovery'},
    '둔화': {'weight': 0.5, 'score': 1, 'color': '#ff7f0e', 'label': 'Slowdown'},
    '침체': {'weight': 0.0, 'score': 0, 'color': '#d62728', 'label': 'Contraction'},
    'Cash': {'weight': 0.0, 'score': -1, 'color': '#ffb347', 'label': 'Cash'},
    'Half': {'weight': 1.0, 'score': -2, 'color': '#9467bd', 'label': 'Half'},
    'Skipped': {'weight': 0.0, 'score': -3, 'color': '#f0f0f0', 'label': 'Skipped'}
}

# =============================================================================
# Core Functions (from v17_PIT.py)
# =============================================================================
def get_next_us_business_day(date: pd.Timestamp) -> pd.Timestamp:
    """
    다음 미국 영업일 반환 (NYSE 캘린더 기준)
    """
    import pandas_market_calendars as mcal
    
    # NYSE 캘린더
    nyse = mcal.get_calendar('NYSE')
    
    # 다음 5일 중 첫 거래일 찾기
    start = date + pd.Timedelta(days=1)
    end = date + pd.Timedelta(days=10)
    trading_days = nyse.schedule(start_date=start, end_date=end).index
    
    if len(trading_days) > 0:
        return pd.Timestamp(trading_days[0])
    else:
        # fallback: 주말만 건너뛰기
        next_day = date + pd.Timedelta(days=1)
        while next_day.weekday() > 4:
            next_day += pd.Timedelta(days=1)
        return next_day


def calculate_eci(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    ECI(Economic Cycle Indicator) 계산
    
    Args:
        df: 'date', 'value' 컬럼이 있는 DataFrame
        
    Returns:
        Level, Momentum, Acceleration, ECI, ECI_Physics가 추가된 DataFrame (또는 None)
    """
    if df is None or len(df) < 13:
        return None

    temp = df.sort_values('date').set_index('date').copy()
    temp = temp[~temp.index.duplicated(keep='last')]

    # 빈 달 채우기
    full_idx = pd.date_range(start=temp.index.min(), end=temp.index.max(), freq='MS')
    temp = temp.reindex(full_idx)
    temp.index.name = 'date'

    # 결측치 채우기 (최대 2달까지)
    temp['value'] = temp['value'].ffill(limit=2)

    # 지표 계산
    temp['ROLLINGAVG'] = temp['value'].rolling(window=12, min_periods=12).mean()
    temp['Level'] = temp['value'] - temp['ROLLINGAVG']
    temp['Momentum'] = temp['Level'].diff()  # 1차 미분 (Velocity)
    temp['Acceleration'] = temp['Momentum'].diff()  # 2차 미분 (Acceleration)

    # 기존 ECI (Level-Momentum 기반)
    conditions = [
        (temp['Level'] > 0) & (temp['Momentum'] > 0),
        (temp['Level'] > 0) & (temp['Momentum'] <= 0),
        (temp['Level'] <= 0) & (temp['Momentum'] > 0)
    ]
    temp['ECI'] = np.select(conditions, ['팽창', '둔화', '회복'], default='침체')
    
    # 새로운 ECI_Physics (Level + Velocity + Acceleration 기반)
    # | Level | Velocity | Acceleration | 해석 | 투자 신호 |
    # | >0    | +        | +            | 팽창 가속 | Strong Buy |
    # | >0    | +        | -            | 팽창 감속 | Hold |
    # | >0    | -        | +            | 둔화 감속 | Watch |
    # | >0    | -        | -            | 둔화 가속 | Exit |
    # | <0    | +        | +            | 회복 가속 | Buy |
    # | <0    | +        | -            | 회복 감속 | Caution |
    # | <0    | -        | +            | 침체 바닥 | Early Entry |
    # | <0    | -        | -            | 침체 가속 | Avoid |
    
    conditions_physics = [
        (temp['Level'] > 0) & (temp['Momentum'] > 0) & (temp['Acceleration'] > 0),   # 팽창 가속
        (temp['Level'] > 0) & (temp['Momentum'] > 0) & (temp['Acceleration'] <= 0),  # 팽창 감속
        (temp['Level'] > 0) & (temp['Momentum'] <= 0) & (temp['Acceleration'] > 0),  # 둔화 감속
        (temp['Level'] > 0) & (temp['Momentum'] <= 0) & (temp['Acceleration'] <= 0), # 둔화 가속
        (temp['Level'] <= 0) & (temp['Momentum'] > 0) & (temp['Acceleration'] > 0),  # 회복 가속
        (temp['Level'] <= 0) & (temp['Momentum'] > 0) & (temp['Acceleration'] <= 0), # 회복 감속
        (temp['Level'] <= 0) & (temp['Momentum'] <= 0) & (temp['Acceleration'] > 0), # 침체 바닥
    ]
    choices_physics = ['팽창++', '팽창+', '둔화-', '둔화--', '회복++', '회복+', '침체-']
    temp['ECI_Physics'] = np.select(conditions_physics, choices_physics, default='침체--')

    # 계산 불가능 구간 NaN 처리
    mask_nan = temp['Level'].isna() | temp['Momentum'].isna() | temp['Acceleration'].isna()
    temp.loc[mask_nan, 'ECI'] = np.nan
    temp.loc[mask_nan, 'ECI_Physics'] = np.nan

    return temp.reset_index()
def get_pit_snapshot(raw_df: pd.DataFrame, obs_date: pd.Timestamp) -> Optional[pd.DataFrame]:
    """
    Point-in-Time 스냅샷: 특정 시점(obs_date)까지 발표된 데이터만 필터링
    
    Args:
        raw_df: 전체 릴리즈 데이터 (realtime_start 컬럼 포함)
        obs_date: 관측 시점
        
    Returns:
        해당 시점에 알 수 있는 데이터만 포함된 DataFrame
    """
    mask = raw_df['realtime_start'] <= obs_date
    known = raw_df[mask]
    if known.empty:
        return None
    return known.sort_values('realtime_start').drop_duplicates(subset=['date'], keep='last').sort_values('date')
def get_score(regime: str) -> int:
    """Regime의 스코어 반환"""
    return REGIME_SETTINGS.get(regime, {'score': -99})['score']
# =============================================================================
# RegimeProvider Class
# =============================================================================
class RegimeProvider:
    """
    Market Regime 데이터 제공자
    
    다른 코드에서 날짜별/국가별 Market Regime을 조회할 수 있는 인터페이스.
    
    Example:
        provider = RegimeProvider(countries=['USA', 'Korea'])
        regime = provider.get_regime('USA', '2024-12-15')
        df = provider.get_regime_series('Korea', '2024-01-01', '2024-12-31')
    """
    
    def __init__(self, 
                 countries: Optional[List[str]] = None, 
                 fred_api_key: Optional[str] = None,
                 auto_load: bool = True,
                 cache_dir: Optional[str] = None,
                 use_cache: bool = True,
                 progress_callback: Optional[Callable[[str, int, int, str], None]] = None):
        """
        초기화
        
        Args:
            countries: 로딩할 국가 리스트. None이면 전체 로딩.
            fred_api_key: FRED API 키. None이면 기본값 사용.
            auto_load: True이면 초기화 시 바로 데이터 로딩
            cache_dir: 캐시 디렉토리 경로. None이면 기본 경로(./cache) 사용.
            use_cache: True이면 일자별 캐시 사용 (같은 날은 API 호출 생략)
            progress_callback: 진행 상황 콜백 함수 (country, current, total, source) -> None
        """
        self.fred_api_key = fred_api_key or '3b56e6990c8059acf92d34b23d723fe5'
        self.countries = countries or list(COUNTRY_MAP.keys())
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.use_cache = use_cache
        self.progress_callback = progress_callback
        
        # 캐시 저장소
        self._raw_data: Dict[str, pd.DataFrame] = {}
        self._first_curve: Dict[str, pd.DataFrame] = {}
        self._first_values: Dict[str, pd.DataFrame] = {}  # For Skipped detection
        self._release_day_map: Dict[str, Dict] = {}  # For Skipped detection
        self._first_regime_map: Dict[str, Dict] = {}
        self._first_vals_map: Dict[str, Dict] = {}
        self._effective_start: Dict[str, pd.Timestamp] = {}
        
        # 이벤트 기반 사전 계산 캐시 (v17_PIT.py와 동일한 결과)
        self._precomputed_regimes: Dict[str, pd.DataFrame] = {}
        
        self._loaded = False
        
        if auto_load:
            self._load_all_data()
    
    def _get_cache_path(self, country: str) -> str:
        """오늘 날짜 기준 캐시 파일 경로 반환"""
        today = datetime.now().strftime('%Y-%m-%d')
        return os.path.join(self.cache_dir, f'{country}_{today}.parquet')
    
    def _load_from_cache(self, country: str) -> Optional[pd.DataFrame]:
        """캐시 파일에서 데이터 로드 (오늘 날짜 기준)"""
        cache_path = self._get_cache_path(country)
        if os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                df['date'] = pd.to_datetime(df['date'])
                df['realtime_start'] = pd.to_datetime(df['realtime_start'])
                return df
            except Exception as e:
                print(f"[WARN] {country} cache load failed: {e}")
        return None
    
    def _save_to_cache(self, country: str, df: pd.DataFrame) -> None:
        """데이터를 캐시 파일로 저장"""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_path = self._get_cache_path(country)
            df.to_parquet(cache_path, index=False)
            print(f"[CACHE] {country} saved: {cache_path}")
        except Exception as e:
            print(f"[WARN] {country} cache save failed: {e}")
    
    def _load_all_data(self):
        """모든 국가 데이터 로딩 (캐시 우선, 없으면 API 호출)"""
        from fredapi import Fred
        fred = None  # 필요할 때만 초기화
        total_countries = len(self.countries)
        
        for i, country in enumerate(self.countries):
            if country not in COUNTRY_MAP:
                print(f"[WARN] Unknown country: {country}")
                continue
            
            info = COUNTRY_MAP[country]
            df_all = None
            source = 'loading'
            
            # Progress callback 호출 - 로딩 시작
            if self.progress_callback:
                self.progress_callback(country, i + 1, total_countries, 'loading')
            
            # 1. 캐시에서 로드 시도
            if self.use_cache:
                df_all = self._load_from_cache(country)
                if df_all is not None:
                    print(f"[CACHE] {country} loaded from cache")
                    source = 'cache'
            
            # 2. 캐시에 없으면 API 호출
            if df_all is None:
                try:
                    if fred is None:
                        fred = Fred(api_key=self.fred_api_key)
                    
                    print(f"[API] {country} fetching from FRED...")
                    source = 'api'
                    df_all = fred.get_series_all_releases(info['fred'])
                    df_all['value'] = pd.to_numeric(df_all['value'], errors='coerce')
                    df_all = df_all.dropna(subset=['value'])
                    df_all['date'] = pd.to_datetime(df_all['date']).dt.tz_localize(None)
                    df_all['realtime_start'] = pd.to_datetime(df_all['realtime_start']).dt.tz_localize(None)
                    
                    # 캐시에 저장
                    if self.use_cache:
                        self._save_to_cache(country, df_all)
                        
                except Exception as e:
                    print(f"[WARN] {country} data loading failed: {e}")
                    continue
            
            self._raw_data[country] = df_all
            
            # First Value Curve 계산
            self._compute_first_curve(country, df_all)
        
        self._loaded = True
    
    def _compute_first_curve(self, country: str, df_all: pd.DataFrame):
        """First Value Curve 사전 계산"""
        df_first_values = df_all.sort_values('realtime_start').drop_duplicates(subset=['date'], keep='first').copy()
        df_first_values['lag'] = (df_first_values['realtime_start'] - df_first_values['date']).dt.days
        df_first_values['is_normal_lag'] = (df_first_values['lag'] >= 30) & (df_first_values['lag'] <= 90)
        df_first_values['prev_normal_1'] = df_first_values['is_normal_lag'].shift(1).fillna(False)
        df_first_values['prev_normal_2'] = df_first_values['is_normal_lag'].shift(2).fillna(False)
        df_first_values['three_consecutive'] = (
            df_first_values['is_normal_lag'] & 
            df_first_values['prev_normal_1'] & 
            df_first_values['prev_normal_2']
        )
        
        # True First Release 시작점 찾기
        true_first_start = None
        consecutive_rows = df_first_values[df_first_values['three_consecutive']]
        if not consecutive_rows.empty:
            first_consecutive_idx = consecutive_rows.index[0]
            first_consecutive_pos = df_first_values.index.get_loc(first_consecutive_idx)
            if first_consecutive_pos > 1:
                true_first_start = df_first_values.iloc[first_consecutive_pos - 2]['date']
            else:
                true_first_start = consecutive_rows.iloc[0]['date']
        
        if true_first_start is not None:
            df_first_values = df_first_values[df_first_values['date'] >= true_first_start]
            
            # Effective Start 계산
            first_release_row = df_first_values[df_first_values['date'] == true_first_start]
            if not first_release_row.empty:
                self._effective_start[country] = first_release_row.iloc[0]['realtime_start']
            else:
                self._effective_start[country] = true_first_start + pd.DateOffset(months=2)
        else:
            self._effective_start[country] = max(
                df_all['realtime_start'].min() + pd.DateOffset(months=24), 
                pd.Timestamp("2000-01-01")
            )
        
        # ECI 계산
        df_first_curve = calculate_eci(df_first_values)
        self._first_curve[country] = df_first_curve
        
        # Skipped 감지용 데이터 저장
        self._first_values[country] = df_first_values
        
        # 발표일 패턴 맵 생성
        df_release_pattern = df_all[['date', 'realtime_start']].drop_duplicates(subset=['date'], keep='first')
        df_release_pattern['release_day'] = df_release_pattern['realtime_start'].dt.day
        self._release_day_map[country] = df_release_pattern.set_index('date')['release_day'].to_dict()
        
        # 빠른 조회용 Dictionary 생성
        if df_first_curve is not None:
            self._first_regime_map[country] = df_first_curve.set_index('date')['ECI'].to_dict()
            self._first_vals_map[country] = df_first_curve.set_index('date')[['Level', 'Momentum']].to_dict('index')
        
        # 이벤트 기반 사전 계산 (v17_PIT.py와 동일)
        self._precompute_event_regimes(country, df_all, df_first_values)
    
    def _precompute_event_regimes(self, country: str, df_all: pd.DataFrame, df_first_values: pd.DataFrame):
        """
        v17_PIT.py와 동일한 이벤트 기반 국면 사전 계산
        실제 발표일 + 동적 체크포인트를 순회하며 결과 저장
        """
        GRACE_PERIOD_DAYS = 2
        effective_start = self._effective_start[country]
        sim_end_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
        
        # 실제 발표일
        actual_releases = df_all[
            (df_all['realtime_start'] >= effective_start) &
            (df_all['realtime_start'] <= sim_end_date)
        ]['realtime_start']
        actual_release_set = set(pd.to_datetime(actual_releases))
        
        # 발표일 패턴 맵
        release_day_map = self._release_day_map[country]
        
        # 동적 체크포인트 생성
        dynamic_checks = []
        check_months = pd.date_range(effective_start, sim_end_date, freq='MS')
        prev_release_day = 15
        
        for month_start in check_months:
            prev_data_month = month_start - pd.DateOffset(months=2)
            if prev_data_month in release_day_map:
                prev_release_day = release_day_map[prev_data_month]
            
            expected_day = min(prev_release_day + GRACE_PERIOD_DAYS, 28)
            try:
                check_date = month_start.replace(day=expected_day)
                while check_date.weekday() > 4:
                    check_date += pd.Timedelta(days=1)
                if check_date <= sim_end_date:
                    dynamic_checks.append(check_date)
            except ValueError:
                pass
        
        # 월말 체크 (안전장치)
        forced_month_end = pd.date_range(effective_start, sim_end_date, freq='BM')
        
        # 모든 이벤트 날짜 병합
        all_dates = pd.concat([
            pd.Series(actual_releases),
            pd.Series(dynamic_checks),
            pd.Series(forced_month_end)
        ]).unique()
        event_dates = sorted(all_dates)
        
        # 이벤트 순회 (v17_PIT.py 로직)
        logs = []
        seen_months = set()
        seen_event_months = set()
        
        for r_date in event_dates:
            r_date = pd.Timestamp(r_date)
            is_actual_release = r_date in actual_release_set
            event_month_key = (r_date.year, r_date.month)
            
            # Snapshot
            snapshot = get_pit_snapshot(df_all, r_date)
            if snapshot is None:
                continue
            
            df_fresh = calculate_eci(snapshot)
            if df_fresh is None or df_fresh.empty:
                continue
            
            latest = df_fresh.iloc[-1]
            m_date = latest['date']
            
            # Missing 판별 (두 가지 조건 중 하나라도 충족하면 Skipped)
            month_diff = (r_date.year * 12 + r_date.month) - (m_date.year * 12 + m_date.month)
            
            # 해당 데이터 월의 실제 lag 확인
            specific_lag = df_first_values[df_first_values['date'] == m_date]['lag'].values
            m_date_release_info = df_first_values[df_first_values['date'] == m_date]
            
            if len(specific_lag) > 0 and not np.isnan(specific_lag[0]):
                import math
                expected_month_diff = max(1, math.ceil(specific_lag[0] / 30))
            else:
                nearby_lags = df_first_values[
                    (df_first_values['date'] >= m_date - pd.DateOffset(months=6)) &
                    (df_first_values['date'] <= m_date + pd.DateOffset(months=6))
                ]['lag'].dropna()
                
                if not nearby_lags.empty:
                    import math
                    avg_lag_months = nearby_lags.median() / 30
                    expected_month_diff = max(1, math.ceil(avg_lag_months))
                else:
                    expected_month_diff = 2
            
            expected_release_day = release_day_map.get(m_date, 15) + GRACE_PERIOD_DAYS
            expected_release_day = min(expected_release_day, 28)
            
            # 조건1: 현재 데이터가 예상보다 1개월 이상 오래됨
            condition1 = (month_diff >= expected_month_diff + 1) and (r_date.day >= expected_release_day)
            
            # 조건2: 마지막 발표 후 30일 이상 지났는데 새 데이터 없음 (월간 발표 기준)
            condition2 = False
            if not m_date_release_info.empty:
                last_release_date = pd.Timestamp(m_date_release_info['realtime_start'].values[0])
                days_since_release = (r_date - last_release_date).days
                # 30일 = 약 1개월 (월간 발표 기준)
                condition2 = days_since_release >= 30
            
            is_missing = condition1 or condition2

            is_new_month = m_date not in seen_months
            if is_new_month:
                seen_months.add(m_date)
            
            # Regime 결정
            m_regime_first = self._first_regime_map.get(country, {}).get(m_date, np.nan)
            m_regime_fresh = latest['ECI']
            
            if is_missing:
                e1_reg = 'Skipped'
                e2_reg = 'Skipped'
                e3_reg = 'Skipped'
                e4_reg = 'Skipped'
            else:
                # Exp1
                if pd.isna(m_regime_first):
                    e1_reg = 'Cash'
                elif not is_new_month:
                    e1_reg = np.nan  # Hold
                else:
                    e1_reg = m_regime_first
                
                # Exp2
                e2_reg = m_regime_fresh if not pd.isna(m_regime_fresh) else 'Cash'
                
                # Exp3
                check_date = m_date - pd.DateOffset(months=1) if is_new_month else m_date
                check_regime_first = self._first_regime_map.get(country, {}).get(check_date, np.nan)
                check_row = df_fresh[df_fresh['date'] == check_date]
                check_regime_fresh = check_row.iloc[0]['ECI'] if not check_row.empty else np.nan
                
                rev_signal = "Neutral"
                if (not pd.isna(check_regime_first)) and (not pd.isna(check_regime_fresh)):
                    s_first = get_score(check_regime_first)
                    s_fresh = get_score(check_regime_fresh)
                    if s_fresh > s_first:
                        rev_signal = "Good"
                    elif s_fresh < s_first:
                        rev_signal = "Bad"
                
                if rev_signal == "Good":
                    e3_reg = m_regime_fresh
                elif rev_signal == "Bad":
                    e3_reg = 'Half' if m_regime_fresh == '팽창' else 'Cash'
                else:
                    e3_reg = m_regime_fresh
                
                # Exp4: Physics-based (Level + Velocity + Acceleration)
                # ECI_Physics 활용하여 세분화된 국면 판단
                m_regime_physics = latest.get('ECI_Physics', np.nan)
                if pd.isna(m_regime_physics):
                    e4_reg = m_regime_fresh  # Fallback to Fresh
                else:
                    # Physics 국면을 투자 신호로 변환
                    # 팽창++, 팽창+ → 팽창
                    # 회복++, 침체- → 회복 (침체 바닥은 Early Entry)
                    # 회복+, 둔화- → 둔화
                    # 둔화--, 침체-- → 침체
                    physics_to_regime = {
                        '팽창++': '팽창',   # Strong Buy
                        '팽창+': '팽창',    # Hold/Buy
                        '회복++': '회복',   # Buy
                        '침체-': '회복',    # Early Entry (침체 바닥)
                        '회복+': '둔화',    # Caution (회복 감속)
                        '둔화-': '둔화',    # Watch (둔화 감속)
                        '둔화--': '침체',   # Exit (둔화 가속)
                        '침체--': '침체',   # Avoid
                    }
                    e4_reg = physics_to_regime.get(m_regime_physics, m_regime_fresh)
            
            # 예상 다음 데이터 월 계산
            next_expected_data_month = m_date + pd.DateOffset(months=1)
            
            # Skipped 사유 기록
            skip_reason = ""
            if is_missing:
                if condition1:
                    skip_reason = f"Expected {next_expected_data_month.strftime('%Y-%m')} data (month_diff={month_diff} >= {expected_month_diff+1})"
                elif condition2:
                    skip_reason = f"Expected {next_expected_data_month.strftime('%Y-%m')} data ({days_since_release}d since last release)"
            
            # 다음 예상 발표일 계산
            if not m_date_release_info.empty:
                last_release = pd.Timestamp(m_date_release_info['realtime_start'].values[0])
                next_expected_release = last_release + pd.Timedelta(days=30)
            else:
                next_expected_release = pd.NaT
            
            # 로그 추가 조건 (v17_PIT.py 로직과 동일)
            # 1) 실제 발표일 이벤트이거나
            # 2) Skipped가 감지된 경우 (체크 전용 이벤트라도 포지션 청산 필요)
            should_log = is_actual_release
            if is_missing:
                should_log = True  # Skipped 감지 시 항상 로그
            
            if should_log:
                seen_event_months.add(event_month_key)
                
                # trade_date 결정: 실제 발표일이면 그 다음 영업일, 
                # Skipped 체크 이벤트이면 해당 월 예상 발표일 기반으로 계산
                if is_actual_release:
                    # 실제 발표일 → 다음 미국 영업일
                    trade_date = get_next_us_business_day(r_date)
                else:
                    # 체크 이벤트에서 Skipped 감지 → 월중순(예상 발표일) 기반으로 계산
                    # 예상 발표일: 해당 월의 예상 발표일 (release_day_map 기반)
                    expected_release_day_for_trade = release_day_map.get(
                        m_date + pd.DateOffset(months=1), 
                        release_day_map.get(m_date, 15)
                    )
                    expected_release_day_for_trade = min(expected_release_day_for_trade, 28)
                    try:
                        # 해당 월의 예상 발표일로 trade_date 설정
                        trade_date = r_date.replace(day=expected_release_day_for_trade)
                        # 주말이면 다음 영업일로
                        while trade_date.weekday() > 4:
                            trade_date += pd.Timedelta(days=1)
                        trade_date = get_next_us_business_day(trade_date - pd.Timedelta(days=1))
                    except ValueError:
                        # 날짜 오류 시 fallback
                        trade_date = get_next_us_business_day(r_date)
                
                logs.append({
                    'trade_date': trade_date,
                    'realtime_start': r_date,
                    'data_month': m_date,
                    'is_missing': is_missing,
                    'skip_reason': skip_reason,
                    'expected_next_data': next_expected_data_month,
                    'expected_next_release': next_expected_release,
                    'exp1_regime': e1_reg,
                    'exp2_regime': e2_reg,
                    'exp3_regime': e3_reg,
                    'exp4_regime': e4_reg,
                    'Level': latest['Level'],
                    'Momentum': latest['Momentum'],
                    'Acceleration': latest.get('Acceleration', np.nan)
                })
        
        if logs:
            df_logs = pd.DataFrame(logs)
            df_logs = df_logs.drop_duplicates('trade_date', keep='last')
            df_logs = df_logs.sort_values('trade_date').reset_index(drop=True)
            self._precomputed_regimes[country] = df_logs
    
    def get_regime(self, 
                   country: str, 
                   date: Union[str, datetime, pd.Timestamp],
                   method: str = 'fresh') -> Optional[Dict]:
        """
        특정 날짜의 국면 조회 (Point-in-Time 기준)
        
        Args:
            country: 국가 코드 (예: 'USA', 'Korea')
            date: 조회 날짜
            method: 'first' (Exp1), 'fresh' (Exp2), 'smart' (Exp3)
            
        Returns:
            {
                'regime': '팽창' | '회복' | '둔화' | '침체' | 'Skipped',
                'Level': float,
                'Momentum': float,
                'data_month': Timestamp (해당하는 데이터 월),
                'weight': float,
                'score': int
            }
            또는 데이터 없으면 None
        """
        if not self._loaded:
            self._load_all_data()
        
        if country not in self._raw_data:
            return None
        
        obs_date = pd.to_datetime(date)
        
        # 사전 계산된 캐시에서 조회 (v17_PIT.py와 동일한 결과)
        precomputed = self._precomputed_regimes.get(country)
        if precomputed is not None and not precomputed.empty:
            # 해당 날짜 이전의 가장 최근 이벤트 찾기
            mask = precomputed['trade_date'] <= obs_date
            if mask.any():
                row = precomputed[mask].iloc[-1]
                
                # method에 따른 regime 선택
                if method == 'first':
                    regime = row['exp1_regime']
                elif method == 'fresh':
                    regime = row['exp2_regime']
                elif method == 'smart':
                    regime = row['exp3_regime']
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                # NaN인 경우 이전 값으로 ffill
                if pd.isna(regime):
                    col = f'exp{"1" if method == "first" else "2" if method == "fresh" else "3"}_regime'
                    filled = precomputed[precomputed['trade_date'] <= obs_date][col].ffill()
                    regime = filled.iloc[-1] if not filled.empty else 'Cash'
                
                if pd.isna(regime):
                    regime = 'Cash'
                
                settings = REGIME_SETTINGS.get(regime, REGIME_SETTINGS['Cash'])
                return {
                    'regime': regime,
                    'Level': row['Level'],
                    'Momentum': row['Momentum'],
                    'data_month': row['data_month'],
                    'weight': settings['weight'],
                    'score': settings['score'],
                    'label': settings['label'],
                    'color': settings['color']
                }
        
        # 캐시에 없으면 동적 계산 (fallback)
        raw_df = self._raw_data[country]
        
        # PIT 스냅샷 생성
        snapshot = get_pit_snapshot(raw_df, obs_date)
        if snapshot is None:
            return None
        
        # Fresh Curve 계산
        df_fresh = calculate_eci(snapshot)
        if df_fresh is None or df_fresh.empty:
            return None
        
        latest = df_fresh.iloc[-1]
        m_date = latest['date']
        
        # Skipped 감지 (데이터 누락 판별)
        is_skipped = self._check_skipped(country, obs_date, m_date)
        if is_skipped:
            settings = REGIME_SETTINGS['Skipped']
            return {
                'regime': 'Skipped',
                'Level': latest['Level'],
                'Momentum': latest['Momentum'],
                'data_month': m_date,
                'weight': settings['weight'],
                'score': settings['score'],
                'label': settings['label'],
                'color': settings['color']
            }
        
        if method == 'first':
            # Exp1: First Value Only
            regime = self._first_regime_map.get(country, {}).get(m_date, np.nan)
            if pd.isna(regime):
                regime = 'Cash'
            vals = self._first_vals_map.get(country, {}).get(m_date, {})
            Level = vals.get('Level', np.nan)
            Momentum = vals.get('Momentum', np.nan)
            
        elif method == 'fresh':
            # Exp2: Fresh Value
            regime = latest['ECI']
            Level = latest['Level']
            Momentum = latest['Momentum']
            
        elif method == 'smart':
            # Exp3: Smart Conditional
            regime, Level, Momentum = self._get_smart_regime(country, m_date, latest, df_fresh)
            
        else:
            raise ValueError(f"Unknown method: {method}. Use 'first', 'fresh', or 'smart'.")
        
        if pd.isna(regime):
            return None
        
        settings = REGIME_SETTINGS.get(regime, REGIME_SETTINGS['Cash'])
        
        return {
            'regime': regime,
            'Level': Level,
            'Momentum': Momentum,
            'data_month': m_date,
            'weight': settings['weight'],
            'score': settings['score'],
            'label': settings['label'],
            'color': settings['color']
        }
    
    def _check_skipped(self, country: str, obs_date: pd.Timestamp, m_date: pd.Timestamp) -> bool:
        """
        데이터 누락(Skipped) 여부 판별 (v17_PIT.py 로직과 동일)
        
        OECD CLI 발표 패턴이 시간에 따라 변경됨:
        - 2018년경: t-2월 데이터 발표 (lag 60~70일)
        - 2022년~: t-1월 데이터 발표 (lag 40~50일)
        동적으로 해당 시기의 정상 lag을 감지하여 skip 판단
        """
        GRACE_PERIOD_DAYS = 2  # 버퍼: 2 영업일
        
        # 월 차이 계산
        month_diff = (obs_date.year * 12 + obs_date.month) - (m_date.year * 12 + m_date.month)
        
        # 해당 국가의 first_values 가져오기
        df_first_values = self._first_values.get(country)
        if df_first_values is None or df_first_values.empty:
            return False
        
        # 최근 데이터의 실제 lag 패턴 확인 (현재 m_date 기준 전후 6개월)
        nearby_lags = df_first_values[
            (df_first_values['date'] >= m_date - pd.DateOffset(months=6)) &
            (df_first_values['date'] <= m_date + pd.DateOffset(months=6))
        ]['lag'].dropna()
        
        if not nearby_lags.empty:
            # 해당 시기의 평균 lag을 월 단위로 변환 (30일 = 1개월)
            avg_lag_months = nearby_lags.median() / 30
            # 정상적인 month_diff 기준 = 평균 lag + 0.5개월 버퍼
            expected_month_diff = int(avg_lag_months + 0.5)
        else:
            expected_month_diff = 1  # 기본값
        
        # 예상 발표일 계산
        release_day_map = self._release_day_map.get(country, {})
        expected_release_day = release_day_map.get(m_date, 15) + GRACE_PERIOD_DAYS
        expected_release_day = min(expected_release_day, 28)
        
        # 누락 판단: 해당 시기 정상 lag보다 1개월 이상 추가 지연되면 skip
        is_missing = (month_diff >= expected_month_diff + 1) and (obs_date.day >= expected_release_day)
        
        return is_missing
    
    def _get_smart_regime(self, country: str, m_date: pd.Timestamp, 
                          latest: pd.Series, df_fresh: pd.DataFrame) -> tuple:
        """Exp3 Smart Conditional 로직"""
        m_regime_fresh = latest['ECI']
        Level = latest['Level']
        Momentum = latest['Momentum']
        
        # 직전 월 비교
        check_date = m_date - pd.DateOffset(months=1)
        check_regime_first = self._first_regime_map.get(country, {}).get(check_date, np.nan)
        check_row = df_fresh[df_fresh['date'] == check_date]
        check_regime_fresh = check_row.iloc[0]['ECI'] if not check_row.empty else np.nan
        
        # 변화 판단
        if (not pd.isna(check_regime_first)) and (not pd.isna(check_regime_fresh)):
            s_first = get_score(check_regime_first)
            s_fresh = get_score(check_regime_fresh)
            
            if s_fresh > s_first:  # Good revision
                return m_regime_fresh, Level, Momentum
            elif s_fresh < s_first:  # Bad revision
                if m_regime_fresh == '팽창':
                    return 'Half', Level, Momentum
                else:
                    return 'Cash', Level, Momentum
        
        return m_regime_fresh, Level, Momentum
    
    def get_regime_series(self,
                          country: str,
                          start: Union[str, datetime, pd.Timestamp],
                          end: Union[str, datetime, pd.Timestamp],
                          method: str = 'fresh',
                          freq: str = 'B') -> pd.DataFrame:
        """
        날짜 범위의 국면 시계열 반환
        
        Args:
            country: 국가 코드
            start: 시작일
            end: 종료일
            method: 'first', 'fresh', 'smart'
            freq: 'B' (영업일), 'D' (일별), 'MS' (월초)
            
        Returns:
            DataFrame with columns: [date, regime, Level, Momentum, weight]
        """
        if not self._loaded:
            self._load_all_data()
        
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        
        dates = pd.date_range(start, end, freq=freq)
        results = []
        
        # 이벤트 날짜 기반으로 계산 (효율성)
        raw_df = self._raw_data.get(country)
        if raw_df is None:
            return pd.DataFrame()
        
        # 해당 기간의 발표일만 추출
        event_dates = raw_df[
            (raw_df['realtime_start'] >= start) & 
            (raw_df['realtime_start'] <= end)
        ]['realtime_start'].unique()
        event_dates = sorted(set(event_dates) | {start, end})
        
        # 각 이벤트 날짜에서 국면 계산
        regime_events = []
        for evt_date in event_dates:
            result = self.get_regime(country, evt_date, method)
            if result:
                regime_events.append({
                    'date': pd.to_datetime(evt_date),
                    **result
                })
        
        if not regime_events:
            return pd.DataFrame()
        
        df_events = pd.DataFrame(regime_events).set_index('date').sort_index()
        
        # 전체 날짜 범위로 확장 (Forward Fill)
        df_daily = df_events.reindex(dates).ffill()
        df_daily.index.name = 'date'
        
        return df_daily.reset_index()
    
    def get_current_regimes(self, method: str = 'fresh') -> Dict[str, str]:
        """
        현재 시점의 전체 국가 국면 요약
        
        Returns:
            {'USA': '팽창', 'Korea': '회복', ...}
        """
        if not self._loaded:
            self._load_all_data()
        
        now = pd.Timestamp.now().normalize()
        result = {}
        
        for country in self.countries:
            regime_info = self.get_regime(country, now, method)
            if regime_info:
                result[country] = regime_info['regime']
            else:
                result[country] = 'N/A'
        
        return result
    
    def get_regime_details(self, country: str, method: str = 'fresh') -> Optional[Dict]:
        """
        현재 시점의 상세 국면 정보
        
        Returns:
            Full regime info dict or None
        """
        return self.get_regime(country, pd.Timestamp.now().normalize(), method)
    
    def refresh(self, country: Optional[str] = None):
        """
        데이터 새로고침
        
        Args:
            country: 특정 국가만 새로고침. None이면 전체.
        """
        if country:
            if country in self._raw_data:
                del self._raw_data[country]
            self.countries = [country]
        else:
            self._raw_data.clear()
            self._first_curve.clear()
            self._first_regime_map.clear()
            self._first_vals_map.clear()
            self._effective_start.clear()
        
        self._loaded = False
        self._load_all_data()
    
    @property
    def available_countries(self) -> List[str]:
        """사용 가능한 국가 목록"""
        return list(COUNTRY_MAP.keys())
    
    @property
    def loaded_countries(self) -> List[str]:
        """로딩된 국가 목록"""
        return list(self._raw_data.keys())
    
    # =========================================================================
    # Visualization Methods
    # =========================================================================
    
    def generate_dashboard(self, 
                           output_path: str = "Regime_Dashboard.html",
                           open_browser: bool = True) -> str:
        """
        v17_PIT.py와 동일한 Bokeh 대시보드 생성
        
        Args:
            output_path: 저장할 HTML 파일 경로
            open_browser: True면 브라우저에서 자동으로 열기
            
        Returns:
            저장된 파일 경로
        """
        from bokeh.plotting import figure, save, output_file
        from bokeh.models import ColumnDataSource, HoverTool, TabPanel, Tabs, Range1d, BoxAnnotation, Arrow, NormalHead
        from bokeh.layouts import column, row
        from bokeh.palettes import Blues256
        from bokeh.resources import INLINE  # file:// 호환성을 위해 INLINE 사용
        import webbrowser
        import os
        
        if not self._loaded:
            self._load_all_data()
        
        # 주가 데이터 로딩
        prices = self._load_price_data()
        
        tabs = []
        regime_colors = {k: v['color'] for k, v in REGIME_SETTINGS.items()}
        
        for country in self.countries:
            if country not in self._raw_data:
                print(f"[DEBUG] {country}: skipped (not in _raw_data)")
                continue
            
            info = COUNTRY_MAP[country]
            c_name = info['name']
            ticker = info['ticker']
            
            # 시계열 데이터 준비
            precomputed = self._precomputed_regimes.get(country)
            if precomputed is None or precomputed.empty:
                print(f"[DEBUG] {country}: skipped (precomputed empty)")
                continue
            
            start_d = self._effective_start.get(country, precomputed['trade_date'].min())
            
            # 차트 생성
            try:
                p_ret = self._make_return_chart(country, c_name, ticker, prices, precomputed, start_d, regime_colors)
                p_strip = self._make_strip_chart(country, precomputed, p_ret.x_range, regime_colors)
                c1, c2, c3 = self._make_clocks(country)
                
                tabs.append(TabPanel(
                    child=column(p_ret, p_strip, row(c1, c2, c3)), 
                    title=c_name
                ))
                print(f"[DEBUG] {country}: tab created successfully")
            except Exception as e:
                print(f"[DEBUG] {country}: FAILED - {e}")
                import traceback
                traceback.print_exc()
        
        if not tabs:
            print("[WARN] No data to visualize")
            return ""
        
        output_file(output_path, title="Market Regime Dashboard", mode="inline")
        save(Tabs(tabs=tabs), resources=INLINE)
        print(f"[OK] Dashboard saved: {output_path}")
        
        if open_browser:
            try:
                webbrowser.open('file://' + os.path.realpath(output_path))
            except:
                pass
        
        return output_path
    
    def _load_price_data(self) -> pd.DataFrame:
        """주가 데이터 로딩 (Yahoo Finance)"""
        try:
            import yfinance as yf
            tickers = [COUNTRY_MAP[c]['ticker'] for c in self.countries if c in COUNTRY_MAP]
            raw_prices = yf.download(tickers, start="1995-01-01", progress=False, auto_adjust=True)
            
            if isinstance(raw_prices.columns, pd.MultiIndex):
                if 'Close' in raw_prices.columns.get_level_values(0):
                    raw_prices = raw_prices.xs('Close', level=0, axis=1)
                elif 'Close' in raw_prices.columns.get_level_values(1):
                    raw_prices = raw_prices.xs('Close', level=1, axis=1)
            
            if isinstance(raw_prices, pd.Series):
                raw_prices = raw_prices.to_frame(name=tickers[0])
            
            if raw_prices.index.tz is not None:
                raw_prices.index = raw_prices.index.tz_localize(None)
            
            return raw_prices.ffill()
        except Exception as e:
            print(f"[WARN] Price data loading failed: {e}")
            return pd.DataFrame()
    
    def _make_return_chart(self, country, c_name, ticker, prices, precomputed, start_d, regime_colors):
        """수익률 차트 생성"""
        from bokeh.plotting import figure
        from bokeh.models import ColumnDataSource
        
        p_ret = figure(title=f"[{c_name}] Cumulative Returns", x_axis_type="datetime",
                       width=1200, height=350, tools="pan,wheel_zoom,reset",
                       y_axis_label="Cumulative Return")
        
        if prices.empty or ticker not in prices.columns:
            return p_ret
        
        bench = prices[ticker].pct_change().fillna(0)
        bench = bench[bench.index >= start_d]
        if bench.empty:
            return p_ret
        
        b_cum = (1 + bench).cumprod()
        b_cum = b_cum / b_cum.iloc[0]
        
        src_b = ColumnDataSource(pd.DataFrame({'date': b_cum.index, 'val': b_cum.values}))
        p_ret.line('date', 'val', source=src_b, color='silver', line_dash='dashed', legend_label='Benchmark')
        
        colors = {'exp1': '#1f77b4', 'exp2': '#2ca02c', 'exp3': '#d62728'}
        labels = {'exp1': 'Exp1 (First)', 'exp2': 'Exp2 (Fresh)', 'exp3': 'Exp3 (Smart)'}
        
        for exp_col in ['exp1', 'exp2', 'exp3']:
            # 일별로 확장
            regime_col = f'{exp_col}_regime'
            if regime_col not in precomputed.columns:
                continue
            
            sub = precomputed[['trade_date', regime_col]].copy()
            sub = sub.set_index('trade_date').reindex(bench.index).ffill()
            sub[regime_col] = sub[regime_col].fillna('Cash')
            
            # Weight 계산
            sub['weight'] = sub[regime_col].map(lambda x: REGIME_SETTINGS.get(x, {'weight': 0})['weight'])
            
            st_ret = (1 + sub['weight'] * bench).cumprod()
            st_ret = st_ret / st_ret.iloc[0]
            
            src_s = ColumnDataSource(pd.DataFrame({
                'date': st_ret.index, 
                'val': st_ret.values, 
                'reg': sub[regime_col]
            }))
            p_ret.line('date', 'val', source=src_s, color=colors[exp_col], line_width=2, legend_label=labels[exp_col])
        
        # 왼쪽 Y축 범위를 수익률 데이터에 맞게 설정
        from bokeh.models import Range1d
        y_min = min(0.5, b_cum.min() * 0.9)  # 최소값
        y_max = max(b_cum.max(), 2.0) * 1.1  # 최대값
        p_ret.y_range = Range1d(start=y_min, end=y_max)
        
        # Crisis-Index 시각화: 우측 y축에 disparity + CX 기반 빨간 음영
        from bokeh.models import LinearAxis, Range1d as Range1d2
        
        crisis_data = self._get_crisis_index(country)
        if crisis_data is not None and not crisis_data.empty and 'disparity' in crisis_data.columns:
            # Regime 시작 시점 이후 데이터만
            crisis_sub = crisis_data[crisis_data.index >= start_d].copy()
            
            if not crisis_sub.empty and 'CX' in crisis_sub.columns:
                # CX 값에 따라 빨간 음영 (CX가 낮을수록 진함)
                from bokeh.models import BoxAnnotation
                
                # 일별로 CX 기반 박스 생성 (성능 위해 변화 시점만)
                crisis_sub['cx_grp'] = (crisis_sub['CX'] != crisis_sub['CX'].shift()).cumsum()
                
                cx_periods = crisis_sub.groupby('cx_grp').agg(
                    start=('CX', lambda x: x.index[0]),
                    end=('CX', lambda x: x.index[-1]),
                    cx_val=('CX', 'first')
                )
                
                for _, row in cx_periods.iterrows():
                    cx = row['cx_val']
                    if pd.notna(cx) and cx < 1.0:  # CX < 1이면 음영
                        # alpha: CX=0이면 0.5, CX=1이면 0
                        alpha = (1 - cx) * 0.5
                        box = BoxAnnotation(left=row['start'], right=row['end'], 
                                           fill_color='#ff6666', fill_alpha=alpha)
                        p_ret.add_layout(box)
                
                # 2) 우측 y축에 disparity 시계열
                disp = crisis_sub['disparity'].dropna()
                if not disp.empty:
                    # 우측 y축 범위 설정
                    y_min, y_max = disp.min() * 0.95, disp.max() * 1.05
                    p_ret.extra_y_ranges = {"disparity": Range1d2(start=y_min, end=y_max)}
                    p_ret.add_layout(LinearAxis(y_range_name="disparity", axis_label="Disparity"), 'right')
                    
                    # disparity 라인
                    src_disp = ColumnDataSource(pd.DataFrame({
                        'date': disp.index,
                        'val': disp.values
                    }))
                    p_ret.line('date', 'val', source=src_disp, y_range_name="disparity",
                              color='#9467bd', line_width=1.5, line_alpha=0.7, legend_label='Disparity')
        
        p_ret.legend.click_policy = "hide"
        p_ret.legend.location = "top_right"
        p_ret.add_layout(p_ret.legend[0], 'right')
        
        return p_ret
    
    def _make_strip_chart(self, country, precomputed, x_range, regime_colors):
        """국면 스트립 차트 생성 (Crisis-Index 포함)"""
        from bokeh.plotting import figure
        from bokeh.models import ColumnDataSource, HoverTool, Range1d
        
        # y축 범위: Exp1/2/3 + Crisis-Index = 5
        p_strip = figure(title=None, x_axis_type="datetime", x_range=x_range, 
                         width=1200, height=180, tools="")
        p_strip.y_range = Range1d(0, 5)
        p_strip.yaxis.ticker = [0.5, 1.5, 2.5, 3.5]
        p_strip.yaxis.major_label_overrides = {0.5: 'Crisis', 1.5: 'Exp3', 2.5: 'Exp2', 3.5: 'Exp1'}
        
        # Exp1/2/3 스트립 (기존 로직, y위치 조정)
        for i, exp_col in enumerate(['exp3_regime', 'exp2_regime', 'exp1_regime']):
            if exp_col not in precomputed.columns:
                continue
            
            sub = precomputed[['trade_date', exp_col]].copy()
            sub = sub.dropna(subset=[exp_col])
            if sub.empty:
                continue
            
            sub['next_date'] = sub['trade_date'].shift(-1).fillna(pd.Timestamp.now())
            y_idx = i + 1  # 1, 2, 3
            
            src = ColumnDataSource(dict(
                d1=sub['trade_date'],
                d2=sub['next_date'],
                bot=[y_idx + 0.1] * len(sub),
                top=[y_idx + 0.9] * len(sub),
                col=sub[exp_col].map(regime_colors),
                reg=sub[exp_col]
            ))
            r = p_strip.quad(left='d1', right='d2', bottom='bot', top='top', color='col', source=src)
            p_strip.add_tools(HoverTool(renderers=[r], tooltips=[
                ('Period', '@d1{%Y-%m}'),
                ('Regime', '@reg')
            ], formatters={'@d1': 'datetime'}))
        
        # Crisis-Index 스트립 (y=0~1) - CX 값 기반 그라데이션
        start_d = self._effective_start.get(country, precomputed['trade_date'].min())
        crisis_data = self._get_crisis_index(country)
        if crisis_data is not None and not crisis_data.empty and 'CX' in crisis_data.columns:
            crisis_sub = crisis_data[crisis_data.index >= start_d][['CX']].dropna()
            if not crisis_sub.empty:
                # CX 변화 시점 기준 그룹핑
                crisis_sub['cx_grp'] = (crisis_sub['CX'] != crisis_sub['CX'].shift()).cumsum()
                crisis_sub = crisis_sub.reset_index()
                crisis_sub.columns = ['date', 'CX', 'cx_grp']
                
                cx_periods = crisis_sub.groupby('cx_grp').agg(
                    start=('date', 'first'),
                    end=('date', 'last'),
                    cx_val=('CX', 'first')
                )
                cx_periods['next_date'] = cx_periods['end'].shift(-1).fillna(pd.Timestamp.now())
                
                # CX 값에 따른 색상: 0=진한 빨강, 0.5=주황, 1=녹색
                def cx_to_color(cx):
                    if pd.isna(cx):
                        return '#cccccc'
                    if cx <= 0:
                        return '#d62728'  # 진한 빨강
                    elif cx < 0.5:
                        return '#ff6666'  # 연한 빨강
                    elif cx < 1.0:
                        return '#ffcc66'  # 주황
                    else:
                        return '#2ca02c'  # 녹색
                
                cx_periods['color'] = cx_periods['cx_val'].apply(cx_to_color)
                
                src_crisis = ColumnDataSource(dict(
                    d1=cx_periods['start'],
                    d2=cx_periods['next_date'],
                    bot=[0.1] * len(cx_periods),
                    top=[0.9] * len(cx_periods),
                    col=cx_periods['color'],
                    cx=cx_periods['cx_val']
                ))
                r_c = p_strip.quad(left='d1', right='d2', bottom='bot', top='top', color='col', source=src_crisis)
                p_strip.add_tools(HoverTool(renderers=[r_c], tooltips=[
                    ('Date', '@d1{%Y-%m-%d}'),
                    ('CX', '@cx{0.2f}')
                ], formatters={'@d1': 'datetime'}))
        
        return p_strip
    
    def _get_crisis_index(self, country: str):
        """
        Crisis-Index (Disparity 기반 buy/sell 신호) 조회
        외부에서 GD_utils 사용 시 override 가능
        """
        # 기본: 캐시된 데이터 반환 (없으면 None)
        if not hasattr(self, '_crisis_cache'):
            self._crisis_cache = {}
        return self._crisis_cache.get(country)
    
    def set_crisis_index(self, country: str, crisis_df: pd.DataFrame):
        """
        외부에서 계산한 Crisis-Index 설정
        
        Args:
            country: 국가 코드
            crisis_df: 'cut' 컬럼이 있는 DataFrame (index=date, cut='buy'/'sell')
        """
        if not hasattr(self, '_crisis_cache'):
            self._crisis_cache = {}
        self._crisis_cache[country] = crisis_df
    
    def _make_clocks(self, country):
        """Business Cycle Clock 3개 생성"""
        first_curve = self._first_curve.get(country)
        precomputed = self._precomputed_regimes.get(country)
        raw_data = self._raw_data.get(country)
        
        # Clock 1: First Value Only
        c1 = self._make_single_clock("1. First Value (Static)", 
                                     first_curve.tail(24) if first_curve is not None else None,
                                     compare=False)
        
        # Clock 2: PIT History (각 발표 시점의 기록)
        if precomputed is not None and not precomputed.empty:
            pit_data = precomputed[['data_month', 'Level', 'Momentum', 'exp2_regime', 'trade_date']].copy()
            pit_data = pit_data.rename(columns={'data_month': 'date', 'exp2_regime': 'ECI'})
            pit_data = pit_data.drop_duplicates(subset=['date'], keep='last')
            
            # First values 추가
            pit_data['Level_first'] = pit_data['date'].map(
                lambda d: self._first_vals_map.get(country, {}).get(d, {}).get('Level', np.nan))
            pit_data['Momentum_first'] = pit_data['date'].map(
                lambda d: self._first_vals_map.get(country, {}).get(d, {}).get('Momentum', np.nan))
            pit_data['Level'] = pit_data['Level']
            pit_data['Momentum'] = pit_data['Momentum']
            
            c2 = self._make_single_clock("2. PIT History (Realized)", pit_data.tail(24), compare=True)
        else:
            c2 = self._make_single_clock("2. PIT History", None, compare=False)
        
        # Clock 3: Current Fresh Snapshot (현재 시점 최신 데이터)
        if raw_data is not None and not raw_data.empty:
            # 현재 시점에서의 fresh ECI 계산
            current_fresh = calculate_eci(raw_data[['date', 'value']].drop_duplicates(subset=['date'], keep='last'))
            if current_fresh is not None and not current_fresh.empty:
                current_fresh = current_fresh.tail(24).copy()
                # First values 추가
                current_fresh['Level_first'] = current_fresh['date'].map(
                    lambda d: self._first_vals_map.get(country, {}).get(d, {}).get('Level', np.nan))
                current_fresh['Momentum_first'] = current_fresh['date'].map(
                    lambda d: self._first_vals_map.get(country, {}).get(d, {}).get('Momentum', np.nan))
                c3 = self._make_single_clock("3. Current Snapshot", current_fresh, compare=True)
            else:
                c3 = self._make_single_clock("3. Current Snapshot", None, compare=False)
        else:
            c3 = self._make_single_clock("3. Current Snapshot", None, compare=False)
        
        return c1, c2, c3
    
    def _make_single_clock(self, title, df, compare=False):
        """단일 Business Cycle Clock 생성"""
        from bokeh.plotting import figure
        from bokeh.models import ColumnDataSource, HoverTool, Range1d, BoxAnnotation, Arrow, NormalHead
        from bokeh.palettes import Blues256
        
        p = figure(title=title, width=380, height=380, tools="pan,wheel_zoom,reset")
        
        if df is None or df.empty or 'Level' not in df.columns:
            axis_range = 4
            p.x_range = Range1d(-axis_range, axis_range)
            p.y_range = Range1d(-axis_range, axis_range)
            return p
        
        d = df.copy()
        
        # 축 범위 계산
        df_valid = d.dropna(subset=['Level', 'Momentum'])
        if df_valid.empty:
            axis_range = 4
        else:
            x_vals = df_valid['Level'].values
            y_vals = df_valid['Momentum'].values
            
            if compare and 'Level_first' in df_valid.columns:
                x_first = df_valid['Level_first'].dropna().values
                y_first = df_valid['Momentum_first'].dropna().values
                if len(x_first) > 0:
                    x_vals = np.concatenate([x_vals, x_first])
                    y_vals = np.concatenate([y_vals, y_first])
            
            x_min, x_max = x_vals.min(), x_vals.max()
            y_min, y_max = y_vals.min(), y_vals.max()
            x_min, x_max = min(x_min, -0.5), max(x_max, 0.5)
            y_min, y_max = min(y_min, -0.5), max(y_max, 0.5)
            
            x_pad = (x_max - x_min) * 0.2
            y_pad = (y_max - y_min) * 0.2
            axis_range = max(abs(x_min - x_pad), abs(x_max + x_pad),
                             abs(y_min - y_pad), abs(y_max + y_pad))
        
        p.x_range = Range1d(-axis_range, axis_range)
        p.y_range = Range1d(-axis_range, axis_range)
        label_pos = axis_range * 0.6
        
        # 사분면 배경
        p.add_layout(BoxAnnotation(bottom=0, left=0, fill_color="#eaffea", fill_alpha=0.3))
        p.add_layout(BoxAnnotation(bottom=0, right=0, fill_color="#fff9c4", fill_alpha=0.3))
        p.add_layout(BoxAnnotation(top=0, right=0, fill_color="#ffe6e6", fill_alpha=0.3))
        p.add_layout(BoxAnnotation(top=0, left=0, fill_color="#fff3e0", fill_alpha=0.3))
        
        # 라벨
        p.text([label_pos, label_pos, -label_pos, -label_pos],
               [label_pos, -label_pos, -label_pos, label_pos],
               text=["Expansion", "Slowdown", "Contraction", "Recovery"],
               text_color=["green", "orange", "red", "gold"], 
               text_align="center", text_font_style="bold", alpha=0.5)
        
        # 비교용 First 포인트
        if compare and 'Level_first' in d.columns:
            valid_first = d.dropna(subset=['Level_first', 'Momentum_first'])
            if not valid_first.empty:
                p.circle(valid_first['Level_first'], valid_first['Momentum_first'], 
                         color='gray', size=4, alpha=0.5, legend_label="First")
                
                # 화살표
                d['dist'] = np.sqrt((d['Level'] - d['Level_first'])**2 + 
                                    (d['Momentum'] - d['Momentum_first'])**2)
                arrows = d[d['dist'] > 0.1].dropna(subset=['Level', 'Momentum', 'Level_first', 'Momentum_first'])
                if not arrows.empty:
                    src_arr = ColumnDataSource(arrows)
                    p.add_layout(Arrow(
                        end=NormalHead(size=5, fill_color="crimson", line_color="crimson"),
                        x_start='Level_first', y_start='Momentum_first',
                        x_end='Level', y_end='Momentum',
                        line_color="crimson", line_dash='dotted', source=src_arr
                    ))
        
        # 경로 그리기
        x, y = d['Level'].values, d['Momentum'].values
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[valid_mask], y[valid_mask]
        
        if len(x) > 1:
            cols = [Blues256[int(50 + i / (len(x) - 1) * 150)] for i in range(len(x) - 1)]
            p.segment(x[:-1], y[:-1], x[1:], y[1:], color=cols, line_width=3)
        
        # 포인트
        if len(x) > 0:
            normal_d = d.dropna(subset=['Level', 'Momentum'])
            src = ColumnDataSource(dict(
                x=normal_d['Level'].values, 
                y=normal_d['Momentum'].values,
                date=normal_d['date'] if 'date' in normal_d.columns else normal_d.index,
                reg=normal_d['ECI'] if 'ECI' in normal_d.columns else ['N/A'] * len(normal_d)
            ))
            c = p.circle('x', 'y', size=8, color='white', line_color='navy', line_width=2, source=src)
            p.circle([x[-1]], [y[-1]], size=12, color='red', legend_label="Latest")
            
            p.add_tools(HoverTool(renderers=[c], tooltips=[
                ("Date", "@date{%Y-%m}"), 
                ("Regime", "@reg")
            ], formatters={'@date': 'datetime'}))
        
        p.legend.location = "bottom_right"
        p.legend.click_policy = "hide"
        return p
# =============================================================================
# 사용 예시 및 검증 (Usage Examples & Validation)
# =============================================================================


if __name__ == '__main__':
    provider = RegimeProvider(use_cache=True)

    # 대시보드 생성 (Crisis-Index 포함)
    provider.generate_dashboard(open_browser=True)

    # AA=provider._precomputed_regimes['Korea']