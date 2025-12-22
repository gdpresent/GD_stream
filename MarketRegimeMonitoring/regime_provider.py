# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import warnings

# GD_utils??ë¡œì»¬ ?˜ê²½?ì„œë§??¬ìš© (Streamlit Cloud?ì„œ??ë¯¸ì‚¬??
try:
    import GD_utils as gdu
except ImportError:
    gdu = None  # Streamlit Cloud ?˜ê²½
from typing import Dict, List, Optional, Union
from datetime import datetime
warnings.filterwarnings('ignore')

# ìºì‹œ ?”ë ‰? ë¦¬ ê¸°ë³¸ ê²½ë¡œ
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')

# =============================================================================
# Configuration
# =============================================================================
COUNTRY_MAP = {
    'USA': {'fred': 'USALOLITOAASTSAM', 'ticker': 'SPY', 'name': 'ë¯¸êµ­(USA)'},
    'Korea': {'fred': 'KORLOLITOAASTSAM', 'ticker': 'EWY', 'name': '?œêµ­(Korea)'},
    'China': {'fred': 'CHNLOLITOAASTSAM', 'ticker': 'MCHI', 'name': 'ì¤‘êµ­(China)'},
    'Japan': {'fred': 'JPNLOLITOAASTSAM', 'ticker': 'EWJ', 'name': '?¼ë³¸(Japan)'},
    'Germany': {'fred': 'DEULOLITOAASTSAM', 'ticker': 'EWG', 'name': '?…ì¼(Germany)'},
    'France': {'fred': 'FRALOLITOAASTSAM', 'ticker': 'EWQ', 'name': '?„ë‘??France)'},
    'UK': {'fred': 'GBRLOLITOAASTSAM', 'ticker': 'EWU', 'name': '?êµ­(UK)'},
    'India': {'fred': 'INDLOLITOAASTSAM', 'ticker': 'INDA', 'name': '?¸ë„(India)'},
    'Brazil': {'fred': 'BRALOLITOAASTSAM', 'ticker': 'EWZ', 'name': 'ë¸Œë¼ì§?Brazil)'},
    'G7': {'fred': 'G7LOLITOAASTSAM', 'ticker': 'ACWI', 'name': 'G7 Global'},
}

# ê°êµ­ ?€?œì???Yahoo Finance ?°ì»¤ (Crisis-Index ??
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
    'G7': '^GSPC',  # G7?€ S&P500 ?€??
}
REGIME_SETTINGS = {
    '?½ì°½': {'weight': 2.0, 'score': 3, 'color': '#2ca02c', 'label': 'Expansion'},
    '?Œë³µ': {'weight': 1.0, 'score': 2, 'color': '#ffce30', 'label': 'Recovery'},
    '?”í™”': {'weight': 0.5, 'score': 1, 'color': '#ff7f0e', 'label': 'Slowdown'},
    'ì¹¨ì²´': {'weight': 0.0, 'score': 0, 'color': '#d62728', 'label': 'Contraction'},
    'Cash': {'weight': 0.0, 'score': -1, 'color': '#ffb347', 'label': 'Cash'},
    'Half': {'weight': 1.0, 'score': -2, 'color': '#9467bd', 'label': 'Half'},
    'Skipped': {'weight': 0.0, 'score': -3, 'color': '#f0f0f0', 'label': 'Skipped'}
}

# =============================================================================
# Core Functions (from v17_PIT.py)
# =============================================================================
def get_next_us_business_day(date: pd.Timestamp) -> pd.Timestamp:
    """
    ?¤ìŒ ë¯¸êµ­ ?ì—…??ë°˜í™˜ (NYSE ìº˜ë¦°??ê¸°ì?)
    """
    import pandas_market_calendars as mcal
    
    # NYSE ìº˜ë¦°??
    nyse = mcal.get_calendar('NYSE')
    
    # ?¤ìŒ 5??ì¤?ì²?ê±°ë˜??ì°¾ê¸°
    start = date + pd.Timedelta(days=1)
    end = date + pd.Timedelta(days=10)
    trading_days = nyse.schedule(start_date=start, end_date=end).index
    
    if len(trading_days) > 0:
        return pd.Timestamp(trading_days[0])
    else:
        # fallback: ì£¼ë§ë§?ê±´ë„ˆ?°ê¸°
        next_day = date + pd.Timedelta(days=1)
        while next_day.weekday() > 4:
            next_day += pd.Timedelta(days=1)
        return next_day


def calculate_eci(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    ECI(Economic Cycle Indicator) ê³„ì‚°
    
    Args:
        df: 'date', 'value' ì»¬ëŸ¼???ˆëŠ” DataFrame
        
    Returns:
        LEVEL, DIRECTION, ECIê°€ ì¶”ê???DataFrame (?ëŠ” None)
    """
    if df is None or len(df) < 13:
        return None

    temp = df.sort_values('date').set_index('date').copy()
    temp = temp[~temp.index.duplicated(keep='last')]

    # ë¹???ì±„ìš°ê¸?
    full_idx = pd.date_range(start=temp.index.min(), end=temp.index.max(), freq='MS')
    temp = temp.reindex(full_idx)
    temp.index.name = 'date'

    # ê²°ì¸¡ì¹?ì±„ìš°ê¸?(ìµœë? 2?¬ê¹Œì§€)
    temp['value'] = temp['value'].ffill(limit=2)

    # ì§€??ê³„ì‚°
    temp['ROLLINGAVG'] = temp['value'].rolling(window=12, min_periods=12).mean()
    temp['LEVEL'] = temp['value'] - temp['ROLLINGAVG']
    temp['DIRECTION'] = temp['LEVEL'].diff()

    conditions = [
        (temp['LEVEL'] > 0) & (temp['DIRECTION'] > 0),
        (temp['LEVEL'] > 0) & (temp['DIRECTION'] <= 0),
        (temp['LEVEL'] <= 0) & (temp['DIRECTION'] > 0)
    ]
    temp['ECI'] = np.select(conditions, ['?½ì°½', '?”í™”', '?Œë³µ'], default='ì¹¨ì²´')

    # ê³„ì‚° ë¶ˆê???êµ¬ê°„ NaN ì²˜ë¦¬
    mask_nan = temp['LEVEL'].isna() | temp['DIRECTION'].isna()
    temp.loc[mask_nan, 'ECI'] = np.nan

    return temp.reset_index()
def get_pit_snapshot(raw_df: pd.DataFrame, obs_date: pd.Timestamp) -> Optional[pd.DataFrame]:
    """
    Point-in-Time ?¤ëƒ…?? ?¹ì • ?œì (obs_date)ê¹Œì? ë°œí‘œ???°ì´?°ë§Œ ?„í„°ë§?
    
    Args:
        raw_df: ?„ì²´ ë¦´ë¦¬ì¦??°ì´??(realtime_start ì»¬ëŸ¼ ?¬í•¨)
        obs_date: ê´€ì¸??œì 
        
    Returns:
        ?´ë‹¹ ?œì ???????ˆëŠ” ?°ì´?°ë§Œ ?¬í•¨??DataFrame
    """
    mask = raw_df['realtime_start'] <= obs_date
    known = raw_df[mask]
    if known.empty:
        return None
    return known.sort_values('realtime_start').drop_duplicates(subset=['date'], keep='last').sort_values('date')
def get_score(regime: str) -> int:
    """Regime???¤ì½”??ë°˜í™˜"""
    return REGIME_SETTINGS.get(regime, {'score': -99})['score']
# =============================================================================
# RegimeProvider Class
# =============================================================================
class RegimeProvider:
    """
    Market Regime ?°ì´???œê³µ??
    
    ?¤ë¥¸ ì½”ë“œ?ì„œ ? ì§œë³?êµ??ë³?Market Regime??ì¡°íšŒ?????ˆëŠ” ?¸í„°?˜ì´??
    
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
                 use_cache: bool = True):
        """
        ì´ˆê¸°??
        
        Args:
            countries: ë¡œë”©??êµ?? ë¦¬ìŠ¤?? None?´ë©´ ?„ì²´ ë¡œë”©.
            fred_api_key: FRED API ?? None?´ë©´ ê¸°ë³¸ê°??¬ìš©.
            auto_load: True?´ë©´ ì´ˆê¸°????ë°”ë¡œ ?°ì´??ë¡œë”©
            cache_dir: ìºì‹œ ?”ë ‰? ë¦¬ ê²½ë¡œ. None?´ë©´ ê¸°ë³¸ ê²½ë¡œ(./cache) ?¬ìš©.
            use_cache: True?´ë©´ ?¼ìë³?ìºì‹œ ?¬ìš© (ê°™ì? ? ì? API ?¸ì¶œ ?ëµ)
        """
        self.fred_api_key = fred_api_key or '3b56e6990c8059acf92d34b23d723fe5'
        self.countries = countries or list(COUNTRY_MAP.keys())
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.use_cache = use_cache
        
        # ìºì‹œ ?€?¥ì†Œ
        self._raw_data: Dict[str, pd.DataFrame] = {}
        self._first_curve: Dict[str, pd.DataFrame] = {}
        self._first_values: Dict[str, pd.DataFrame] = {}  # For Skipped detection
        self._release_day_map: Dict[str, Dict] = {}  # For Skipped detection
        self._first_regime_map: Dict[str, Dict] = {}
        self._first_vals_map: Dict[str, Dict] = {}
        self._effective_start: Dict[str, pd.Timestamp] = {}
        
        # ?´ë²¤??ê¸°ë°˜ ?¬ì „ ê³„ì‚° ìºì‹œ (v17_PIT.py?€ ?™ì¼??ê²°ê³¼)
        self._precomputed_regimes: Dict[str, pd.DataFrame] = {}
        
        self._loaded = False
        
        if auto_load:
            self._load_all_data()
    
    def _get_cache_path(self, country: str) -> str:
        """?¤ëŠ˜ ? ì§œ ê¸°ì? ìºì‹œ ?Œì¼ ê²½ë¡œ ë°˜í™˜"""
        today = datetime.now().strftime('%Y-%m-%d')
        return os.path.join(self.cache_dir, f'{country}_{today}.parquet')
    
    def _load_from_cache(self, country: str) -> Optional[pd.DataFrame]:
        """ìºì‹œ ?Œì¼?ì„œ ?°ì´??ë¡œë“œ (?¤ëŠ˜ ? ì§œ ê¸°ì?)"""
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
        """?°ì´?°ë? ìºì‹œ ?Œì¼ë¡??€??""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_path = self._get_cache_path(country)
            df.to_parquet(cache_path, index=False)
            print(f"[CACHE] {country} saved: {cache_path}")
        except Exception as e:
            print(f"[WARN] {country} cache save failed: {e}")
    
    def _load_all_data(self):
        """ëª¨ë“  êµ?? ?°ì´??ë¡œë”© (ìºì‹œ ?°ì„ , ?†ìœ¼ë©?API ?¸ì¶œ)"""
        from fredapi import Fred
        fred = None  # ?„ìš”???Œë§Œ ì´ˆê¸°??
        
        for country in self.countries:
            if country not in COUNTRY_MAP:
                print(f"[WARN] Unknown country: {country}")
                continue
            
            info = COUNTRY_MAP[country]
            df_all = None
            
            # 1. ìºì‹œ?ì„œ ë¡œë“œ ?œë„
            if self.use_cache:
                df_all = self._load_from_cache(country)
                if df_all is not None:
                    print(f"[CACHE] {country} loaded from cache")
            
            # 2. ìºì‹œ???†ìœ¼ë©?API ?¸ì¶œ
            if df_all is None:
                try:
                    if fred is None:
                        fred = Fred(api_key=self.fred_api_key)
                    
                    print(f"[API] {country} fetching from FRED...")
                    df_all = fred.get_series_all_releases(info['fred'])
                    df_all['value'] = pd.to_numeric(df_all['value'], errors='coerce')
                    df_all = df_all.dropna(subset=['value'])
                    df_all['date'] = pd.to_datetime(df_all['date']).dt.tz_localize(None)
                    df_all['realtime_start'] = pd.to_datetime(df_all['realtime_start']).dt.tz_localize(None)
                    
                    # ìºì‹œ???€??
                    if self.use_cache:
                        self._save_to_cache(country, df_all)
                        
                except Exception as e:
                    print(f"[WARN] {country} data loading failed: {e}")
                    continue
            
            self._raw_data[country] = df_all
            
            # First Value Curve ê³„ì‚°
            self._compute_first_curve(country, df_all)
        
        self._loaded = True
    
    def _compute_first_curve(self, country: str, df_all: pd.DataFrame):
        """First Value Curve ?¬ì „ ê³„ì‚°"""
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
        
        # True First Release ?œì‘??ì°¾ê¸°
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
            
            # Effective Start ê³„ì‚°
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
        
        # ECI ê³„ì‚°
        df_first_curve = calculate_eci(df_first_values)
        self._first_curve[country] = df_first_curve
        
        # Skipped ê°ì????°ì´???€??
        self._first_values[country] = df_first_values
        
        # ë°œí‘œ???¨í„´ ë§??ì„±
        df_release_pattern = df_all[['date', 'realtime_start']].drop_duplicates(subset=['date'], keep='first')
        df_release_pattern['release_day'] = df_release_pattern['realtime_start'].dt.day
        self._release_day_map[country] = df_release_pattern.set_index('date')['release_day'].to_dict()
        
        # ë¹ ë¥¸ ì¡°íšŒ??Dictionary ?ì„±
        if df_first_curve is not None:
            self._first_regime_map[country] = df_first_curve.set_index('date')['ECI'].to_dict()
            self._first_vals_map[country] = df_first_curve.set_index('date')[['LEVEL', 'DIRECTION']].to_dict('index')
        
        # ?´ë²¤??ê¸°ë°˜ ?¬ì „ ê³„ì‚° (v17_PIT.py?€ ?™ì¼)
        self._precompute_event_regimes(country, df_all, df_first_values)
    
    def _precompute_event_regimes(self, country: str, df_all: pd.DataFrame, df_first_values: pd.DataFrame):
        """
        v17_PIT.py?€ ?™ì¼???´ë²¤??ê¸°ë°˜ êµ?©´ ?¬ì „ ê³„ì‚°
        ?¤ì œ ë°œí‘œ??+ ?™ì  ì²´í¬?¬ì¸?¸ë? ?œíšŒ?˜ë©° ê²°ê³¼ ?€??
        """
        GRACE_PERIOD_DAYS = 2
        effective_start = self._effective_start[country]
        sim_end_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
        
        # ?¤ì œ ë°œí‘œ??
        actual_releases = df_all[
            (df_all['realtime_start'] >= effective_start) &
            (df_all['realtime_start'] <= sim_end_date)
        ]['realtime_start']
        actual_release_set = set(pd.to_datetime(actual_releases))
        
        # ë°œí‘œ???¨í„´ ë§?
        release_day_map = self._release_day_map[country]
        
        # ?™ì  ì²´í¬?¬ì¸???ì„±
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
        
        # ?”ë§ ì²´í¬ (?ˆì „?¥ì¹˜)
        forced_month_end = pd.date_range(effective_start, sim_end_date, freq='BM')
        
        # ëª¨ë“  ?´ë²¤??? ì§œ ë³‘í•©
        all_dates = pd.concat([
            pd.Series(actual_releases),
            pd.Series(dynamic_checks),
            pd.Series(forced_month_end)
        ]).unique()
        event_dates = sorted(all_dates)
        
        # ?´ë²¤???œíšŒ (v17_PIT.py ë¡œì§)
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
            
            # Missing ?ë³„ (??ê°€ì§€ ì¡°ê±´ ì¤??˜ë‚˜?¼ë„ ì¶©ì¡±?˜ë©´ Skipped)
            month_diff = (r_date.year * 12 + r_date.month) - (m_date.year * 12 + m_date.month)
            
            # ?´ë‹¹ ?°ì´???”ì˜ ?¤ì œ lag ?•ì¸
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
            
            # ì¡°ê±´1: ?„ì¬ ?°ì´?°ê? ?ˆìƒë³´ë‹¤ 1ê°œì›” ?´ìƒ ?¤ë˜??
            condition1 = (month_diff >= expected_month_diff + 1) and (r_date.day >= expected_release_day)
            
            # ì¡°ê±´2: ë§ˆì?ë§?ë°œí‘œ ??30???´ìƒ ì§€?¬ëŠ”?????°ì´???†ìŒ (?”ê°„ ë°œí‘œ ê¸°ì?)
            condition2 = False
            if not m_date_release_info.empty:
                last_release_date = pd.Timestamp(m_date_release_info['realtime_start'].values[0])
                days_since_release = (r_date - last_release_date).days
                # 30??= ??1ê°œì›” (?”ê°„ ë°œí‘œ ê¸°ì?)
                condition2 = days_since_release >= 30
            
            is_missing = condition1 or condition2

            is_new_month = m_date not in seen_months
            if is_new_month:
                seen_months.add(m_date)
            
            # Regime ê²°ì •
            m_regime_first = self._first_regime_map.get(country, {}).get(m_date, np.nan)
            m_regime_fresh = latest['ECI']
            
            if is_missing:
                e1_reg = 'Skipped'
                e2_reg = 'Skipped'
                e3_reg = 'Skipped'
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
                    e3_reg = 'Half' if m_regime_fresh == '?½ì°½' else 'Cash'
                else:
                    e3_reg = m_regime_fresh
            
            # ?ˆìƒ ?¤ìŒ ?°ì´????ê³„ì‚°
            next_expected_data_month = m_date + pd.DateOffset(months=1)
            
            # Skipped ?¬ìœ  ê¸°ë¡
            skip_reason = ""
            if is_missing:
                if condition1:
                    skip_reason = f"Expected {next_expected_data_month.strftime('%Y-%m')} data (month_diff={month_diff} >= {expected_month_diff+1})"
                elif condition2:
                    skip_reason = f"Expected {next_expected_data_month.strftime('%Y-%m')} data ({days_since_release}d since last release)"
            
            # ?¤ìŒ ?ˆìƒ ë°œí‘œ??ê³„ì‚°
            if not m_date_release_info.empty:
                last_release = pd.Timestamp(m_date_release_info['realtime_start'].values[0])
                next_expected_release = last_release + pd.Timedelta(days=30)
            else:
                next_expected_release = pd.NaT
            
            # ë¡œê·¸ ì¶”ê? ì¡°ê±´ (v17_PIT.py ë¡œì§ê³??™ì¼)
            # 1) ?¤ì œ ë°œí‘œ???´ë²¤?¸ì´ê±°ë‚˜
            # 2) Skippedê°€ ê°ì???ê²½ìš° (ì²´í¬ ?„ìš© ?´ë²¤?¸ë¼???¬ì???ì²?‚° ?„ìš”)
            should_log = is_actual_release
            if is_missing:
                should_log = True  # Skipped ê°ì? ????ƒ ë¡œê·¸
            
            if should_log:
                seen_event_months.add(event_month_key)
                # ?¤ìŒ ë¯¸êµ­ ?ì—…??ê³„ì‚° (ì£¼ë§ + ?´ì¼ ?œì™¸)
                next_day = get_next_us_business_day(r_date)
                logs.append({
                    'trade_date': next_day,
                    'realtime_start': r_date,
                    'data_month': m_date,
                    'is_missing': is_missing,
                    'skip_reason': skip_reason,
                    'expected_next_data': next_expected_data_month,
                    'expected_next_release': next_expected_release,
                    'exp1_regime': e1_reg,
                    'exp2_regime': e2_reg,
                    'exp3_regime': e3_reg,
                    'LEVEL': latest['LEVEL'],
                    'DIRECTION': latest['DIRECTION']
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
        ?¹ì • ? ì§œ??êµ?©´ ì¡°íšŒ (Point-in-Time ê¸°ì?)
        
        Args:
            country: êµ?? ì½”ë“œ (?? 'USA', 'Korea')
            date: ì¡°íšŒ ? ì§œ
            method: 'first' (Exp1), 'fresh' (Exp2), 'smart' (Exp3)
            
        Returns:
            {
                'regime': '?½ì°½' | '?Œë³µ' | '?”í™”' | 'ì¹¨ì²´' | 'Skipped',
                'LEVEL': float,
                'DIRECTION': float,
                'data_month': Timestamp (?´ë‹¹?˜ëŠ” ?°ì´????,
                'weight': float,
                'score': int
            }
            ?ëŠ” ?°ì´???†ìœ¼ë©?None
        """
        if not self._loaded:
            self._load_all_data()
        
        if country not in self._raw_data:
            return None
        
        obs_date = pd.to_datetime(date)
        
        # ?¬ì „ ê³„ì‚°??ìºì‹œ?ì„œ ì¡°íšŒ (v17_PIT.py?€ ?™ì¼??ê²°ê³¼)
        precomputed = self._precomputed_regimes.get(country)
        if precomputed is not None and not precomputed.empty:
            # ?´ë‹¹ ? ì§œ ?´ì „??ê°€??ìµœê·¼ ?´ë²¤??ì°¾ê¸°
            mask = precomputed['trade_date'] <= obs_date
            if mask.any():
                row = precomputed[mask].iloc[-1]
                
                # method???°ë¥¸ regime ? íƒ
                if method == 'first':
                    regime = row['exp1_regime']
                elif method == 'fresh':
                    regime = row['exp2_regime']
                elif method == 'smart':
                    regime = row['exp3_regime']
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                # NaN??ê²½ìš° ?´ì „ ê°’ìœ¼ë¡?ffill
                if pd.isna(regime):
                    col = f'exp{"1" if method == "first" else "2" if method == "fresh" else "3"}_regime'
                    filled = precomputed[precomputed['trade_date'] <= obs_date][col].ffill()
                    regime = filled.iloc[-1] if not filled.empty else 'Cash'
                
                if pd.isna(regime):
                    regime = 'Cash'
                
                settings = REGIME_SETTINGS.get(regime, REGIME_SETTINGS['Cash'])
                return {
                    'regime': regime,
                    'LEVEL': row['LEVEL'],
                    'DIRECTION': row['DIRECTION'],
                    'data_month': row['data_month'],
                    'weight': settings['weight'],
                    'score': settings['score'],
                    'label': settings['label'],
                    'color': settings['color']
                }
        
        # ìºì‹œ???†ìœ¼ë©??™ì  ê³„ì‚° (fallback)
        raw_df = self._raw_data[country]
        
        # PIT ?¤ëƒ…???ì„±
        snapshot = get_pit_snapshot(raw_df, obs_date)
        if snapshot is None:
            return None
        
        # Fresh Curve ê³„ì‚°
        df_fresh = calculate_eci(snapshot)
        if df_fresh is None or df_fresh.empty:
            return None
        
        latest = df_fresh.iloc[-1]
        m_date = latest['date']
        
        # Skipped ê°ì? (?°ì´???„ë½ ?ë³„)
        is_skipped = self._check_skipped(country, obs_date, m_date)
        if is_skipped:
            settings = REGIME_SETTINGS['Skipped']
            return {
                'regime': 'Skipped',
                'LEVEL': latest['LEVEL'],
                'DIRECTION': latest['DIRECTION'],
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
            LEVEL = vals.get('LEVEL', np.nan)
            DIRECTION = vals.get('DIRECTION', np.nan)
            
        elif method == 'fresh':
            # Exp2: Fresh Value
            regime = latest['ECI']
            LEVEL = latest['LEVEL']
            DIRECTION = latest['DIRECTION']
            
        elif method == 'smart':
            # Exp3: Smart Conditional
            regime, LEVEL, DIRECTION = self._get_smart_regime(country, m_date, latest, df_fresh)
            
        else:
            raise ValueError(f"Unknown method: {method}. Use 'first', 'fresh', or 'smart'.")
        
        if pd.isna(regime):
            return None
        
        settings = REGIME_SETTINGS.get(regime, REGIME_SETTINGS['Cash'])
        
        return {
            'regime': regime,
            'LEVEL': LEVEL,
            'DIRECTION': DIRECTION,
            'data_month': m_date,
            'weight': settings['weight'],
            'score': settings['score'],
            'label': settings['label'],
            'color': settings['color']
        }
    
    def _check_skipped(self, country: str, obs_date: pd.Timestamp, m_date: pd.Timestamp) -> bool:
        """
        ?°ì´???„ë½(Skipped) ?¬ë? ?ë³„ (v17_PIT.py ë¡œì§ê³??™ì¼)
        
        OECD CLI ë°œí‘œ ?¨í„´???œê°„???°ë¼ ë³€ê²½ë¨:
        - 2018?„ê²½: t-2???°ì´??ë°œí‘œ (lag 60~70??
        - 2022??: t-1???°ì´??ë°œí‘œ (lag 40~50??
        ?™ì ?¼ë¡œ ?´ë‹¹ ?œê¸°???•ìƒ lag??ê°ì??˜ì—¬ skip ?ë‹¨
        """
        GRACE_PERIOD_DAYS = 2  # ë²„í¼: 2 ?ì—…??
        
        # ??ì°¨ì´ ê³„ì‚°
        month_diff = (obs_date.year * 12 + obs_date.month) - (m_date.year * 12 + m_date.month)
        
        # ?´ë‹¹ êµ????first_values ê°€?¸ì˜¤ê¸?
        df_first_values = self._first_values.get(country)
        if df_first_values is None or df_first_values.empty:
            return False
        
        # ìµœê·¼ ?°ì´?°ì˜ ?¤ì œ lag ?¨í„´ ?•ì¸ (?„ì¬ m_date ê¸°ì? ?„í›„ 6ê°œì›”)
        nearby_lags = df_first_values[
            (df_first_values['date'] >= m_date - pd.DateOffset(months=6)) &
            (df_first_values['date'] <= m_date + pd.DateOffset(months=6))
        ]['lag'].dropna()
        
        if not nearby_lags.empty:
            # ?´ë‹¹ ?œê¸°???‰ê·  lag?????¨ìœ„ë¡?ë³€??(30??= 1ê°œì›”)
            avg_lag_months = nearby_lags.median() / 30
            # ?•ìƒ?ì¸ month_diff ê¸°ì? = ?‰ê·  lag + 0.5ê°œì›” ë²„í¼
            expected_month_diff = int(avg_lag_months + 0.5)
        else:
            expected_month_diff = 1  # ê¸°ë³¸ê°?
        
        # ?ˆìƒ ë°œí‘œ??ê³„ì‚°
        release_day_map = self._release_day_map.get(country, {})
        expected_release_day = release_day_map.get(m_date, 15) + GRACE_PERIOD_DAYS
        expected_release_day = min(expected_release_day, 28)
        
        # ?„ë½ ?ë‹¨: ?´ë‹¹ ?œê¸° ?•ìƒ lagë³´ë‹¤ 1ê°œì›” ?´ìƒ ì¶”ê? ì§€?°ë˜ë©?skip
        is_missing = (month_diff >= expected_month_diff + 1) and (obs_date.day >= expected_release_day)
        
        return is_missing
    
    def _get_smart_regime(self, country: str, m_date: pd.Timestamp, 
                          latest: pd.Series, df_fresh: pd.DataFrame) -> tuple:
        """Exp3 Smart Conditional ë¡œì§"""
        m_regime_fresh = latest['ECI']
        LEVEL = latest['LEVEL']
        DIRECTION = latest['DIRECTION']
        
        # ì§ì „ ??ë¹„êµ
        check_date = m_date - pd.DateOffset(months=1)
        check_regime_first = self._first_regime_map.get(country, {}).get(check_date, np.nan)
        check_row = df_fresh[df_fresh['date'] == check_date]
        check_regime_fresh = check_row.iloc[0]['ECI'] if not check_row.empty else np.nan
        
        # ë³€???ë‹¨
        if (not pd.isna(check_regime_first)) and (not pd.isna(check_regime_fresh)):
            s_first = get_score(check_regime_first)
            s_fresh = get_score(check_regime_fresh)
            
            if s_fresh > s_first:  # Good revision
                return m_regime_fresh, LEVEL, DIRECTION
            elif s_fresh < s_first:  # Bad revision
                if m_regime_fresh == '?½ì°½':
                    return 'Half', LEVEL, DIRECTION
                else:
                    return 'Cash', LEVEL, DIRECTION
        
        return m_regime_fresh, LEVEL, DIRECTION
    
    def get_regime_series(self,
                          country: str,
                          start: Union[str, datetime, pd.Timestamp],
                          end: Union[str, datetime, pd.Timestamp],
                          method: str = 'fresh',
                          freq: str = 'B') -> pd.DataFrame:
        """
        ? ì§œ ë²”ìœ„??êµ?©´ ?œê³„??ë°˜í™˜
        
        Args:
            country: êµ?? ì½”ë“œ
            start: ?œì‘??
            end: ì¢…ë£Œ??
            method: 'first', 'fresh', 'smart'
            freq: 'B' (?ì—…??, 'D' (?¼ë³„), 'MS' (?”ì´ˆ)
            
        Returns:
            DataFrame with columns: [date, regime, LEVEL, DIRECTION, weight]
        """
        if not self._loaded:
            self._load_all_data()
        
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        
        dates = pd.date_range(start, end, freq=freq)
        results = []
        
        # ?´ë²¤??? ì§œ ê¸°ë°˜?¼ë¡œ ê³„ì‚° (?¨ìœ¨??
        raw_df = self._raw_data.get(country)
        if raw_df is None:
            return pd.DataFrame()
        
        # ?´ë‹¹ ê¸°ê°„??ë°œí‘œ?¼ë§Œ ì¶”ì¶œ
        event_dates = raw_df[
            (raw_df['realtime_start'] >= start) & 
            (raw_df['realtime_start'] <= end)
        ]['realtime_start'].unique()
        event_dates = sorted(set(event_dates) | {start, end})
        
        # ê°??´ë²¤??? ì§œ?ì„œ êµ?©´ ê³„ì‚°
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
        
        # ?„ì²´ ? ì§œ ë²”ìœ„ë¡??•ì¥ (Forward Fill)
        df_daily = df_events.reindex(dates).ffill()
        df_daily.index.name = 'date'
        
        return df_daily.reset_index()
    
    def get_current_regimes(self, method: str = 'fresh') -> Dict[str, str]:
        """
        ?„ì¬ ?œì ???„ì²´ êµ?? êµ?©´ ?”ì•½
        
        Returns:
            {'USA': '?½ì°½', 'Korea': '?Œë³µ', ...}
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
        ?„ì¬ ?œì ???ì„¸ êµ?©´ ?•ë³´
        
        Returns:
            Full regime info dict or None
        """
        return self.get_regime(country, pd.Timestamp.now().normalize(), method)
    
    def refresh(self, country: Optional[str] = None):
        """
        ?°ì´???ˆë¡œê³ ì¹¨
        
        Args:
            country: ?¹ì • êµ??ë§??ˆë¡œê³ ì¹¨. None?´ë©´ ?„ì²´.
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
        """?¬ìš© ê°€?¥í•œ êµ?? ëª©ë¡"""
        return list(COUNTRY_MAP.keys())
    
    @property
    def loaded_countries(self) -> List[str]:
        """ë¡œë”©??êµ?? ëª©ë¡"""
        return list(self._raw_data.keys())
    
    # =========================================================================
    # Visualization Methods
    # =========================================================================
    
    def generate_dashboard(self, 
                           output_path: str = "Regime_Dashboard.html",
                           open_browser: bool = True) -> str:
        """
        v17_PIT.py?€ ?™ì¼??Bokeh ?€?œë³´???ì„±
        
        Args:
            output_path: ?€?¥í•  HTML ?Œì¼ ê²½ë¡œ
            open_browser: Trueë©?ë¸Œë¼?°ì??ì„œ ?ë™?¼ë¡œ ?´ê¸°
            
        Returns:
            ?€?¥ëœ ?Œì¼ ê²½ë¡œ
        """
        from bokeh.plotting import figure, save, output_file
        from bokeh.models import ColumnDataSource, HoverTool, TabPanel, Tabs, Range1d, BoxAnnotation, Arrow, NormalHead
        from bokeh.layouts import column, row
        from bokeh.palettes import Blues256
        from bokeh.resources import INLINE  # file:// ?¸í™˜?±ì„ ?„í•´ INLINE ?¬ìš©
        import webbrowser
        import os
        
        if not self._loaded:
            self._load_all_data()
        
        # ì£¼ê? ?°ì´??ë¡œë”©
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
            
            # ?œê³„???°ì´??ì¤€ë¹?
            precomputed = self._precomputed_regimes.get(country)
            if precomputed is None or precomputed.empty:
                print(f"[DEBUG] {country}: skipped (precomputed empty)")
                continue
            
            start_d = self._effective_start.get(country, precomputed['trade_date'].min())
            
            # ì°¨íŠ¸ ?ì„±
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
        """ì£¼ê? ?°ì´??ë¡œë”© (Yahoo Finance)"""
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
        """?˜ìµë¥?ì°¨íŠ¸ ?ì„±"""
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
            # ?¼ë³„ë¡??•ì¥
            regime_col = f'{exp_col}_regime'
            if regime_col not in precomputed.columns:
                continue
            
            sub = precomputed[['trade_date', regime_col]].copy()
            sub = sub.set_index('trade_date').reindex(bench.index).ffill()
            sub[regime_col] = sub[regime_col].fillna('Cash')
            
            # Weight ê³„ì‚°
            sub['weight'] = sub[regime_col].map(lambda x: REGIME_SETTINGS.get(x, {'weight': 0})['weight'])
            
            st_ret = (1 + sub['weight'] * bench).cumprod()
            st_ret = st_ret / st_ret.iloc[0]
            
            src_s = ColumnDataSource(pd.DataFrame({
                'date': st_ret.index, 
                'val': st_ret.values, 
                'reg': sub[regime_col]
            }))
            p_ret.line('date', 'val', source=src_s, color=colors[exp_col], line_width=2, legend_label=labels[exp_col])
        
        # ?¼ìª½ Yì¶?ë²”ìœ„ë¥??˜ìµë¥??°ì´?°ì— ë§ê²Œ ?¤ì •
        from bokeh.models import Range1d
        y_min = min(0.5, b_cum.min() * 0.9)  # ìµœì†Œê°?
        y_max = max(b_cum.max(), 2.0) * 1.1  # ìµœë?ê°?
        p_ret.y_range = Range1d(start=y_min, end=y_max)
        
        # Crisis-Index ?œê°?? ?°ì¸¡ yì¶•ì— disparity + CX ê¸°ë°˜ ë¹¨ê°„ ?Œì˜
        from bokeh.models import LinearAxis, Range1d as Range1d2
        
        crisis_data = self._get_crisis_index(country)
        if crisis_data is not None and not crisis_data.empty and 'disparity' in crisis_data.columns:
            # Regime ?œì‘ ?œì  ?´í›„ ?°ì´?°ë§Œ
            crisis_sub = crisis_data[crisis_data.index >= start_d].copy()
            
            if not crisis_sub.empty and 'CX' in crisis_sub.columns:
                # CX ê°’ì— ?°ë¼ ë¹¨ê°„ ?Œì˜ (CXê°€ ??„?˜ë¡ ì§„í•¨)
                from bokeh.models import BoxAnnotation
                
                # ?¼ë³„ë¡?CX ê¸°ë°˜ ë°•ìŠ¤ ?ì„± (?±ëŠ¥ ?„í•´ ë³€???œì ë§?
                crisis_sub['cx_grp'] = (crisis_sub['CX'] != crisis_sub['CX'].shift()).cumsum()
                
                cx_periods = crisis_sub.groupby('cx_grp').agg(
                    start=('CX', lambda x: x.index[0]),
                    end=('CX', lambda x: x.index[-1]),
                    cx_val=('CX', 'first')
                )
                
                for _, row in cx_periods.iterrows():
                    cx = row['cx_val']
                    if pd.notna(cx) and cx < 1.0:  # CX < 1?´ë©´ ?Œì˜
                        # alpha: CX=0?´ë©´ 0.5, CX=1?´ë©´ 0
                        alpha = (1 - cx) * 0.5
                        box = BoxAnnotation(left=row['start'], right=row['end'], 
                                           fill_color='#ff6666', fill_alpha=alpha)
                        p_ret.add_layout(box)
                
                # 2) ?°ì¸¡ yì¶•ì— disparity ?œê³„??
                disp = crisis_sub['disparity'].dropna()
                if not disp.empty:
                    # ?°ì¸¡ yì¶?ë²”ìœ„ ?¤ì •
                    y_min, y_max = disp.min() * 0.95, disp.max() * 1.05
                    p_ret.extra_y_ranges = {"disparity": Range1d2(start=y_min, end=y_max)}
                    p_ret.add_layout(LinearAxis(y_range_name="disparity", axis_label="Disparity"), 'right')
                    
                    # disparity ?¼ì¸
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
        """êµ?©´ ?¤íŠ¸ë¦?ì°¨íŠ¸ ?ì„± (Crisis-Index ?¬í•¨)"""
        from bokeh.plotting import figure
        from bokeh.models import ColumnDataSource, HoverTool, Range1d
        
        # yì¶?ë²”ìœ„: Exp1/2/3 + Crisis-Index = 5
        p_strip = figure(title=None, x_axis_type="datetime", x_range=x_range, 
                         width=1200, height=180, tools="")
        p_strip.y_range = Range1d(0, 5)
        p_strip.yaxis.ticker = [0.5, 1.5, 2.5, 3.5]
        p_strip.yaxis.major_label_overrides = {0.5: 'Crisis', 1.5: 'Exp3', 2.5: 'Exp2', 3.5: 'Exp1'}
        
        # Exp1/2/3 ?¤íŠ¸ë¦?(ê¸°ì¡´ ë¡œì§, y?„ì¹˜ ì¡°ì •)
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
        
        # Crisis-Index ?¤íŠ¸ë¦?(y=0~1) - CX ê°?ê¸°ë°˜ ê·¸ë¼?°ì´??
        start_d = self._effective_start.get(country, precomputed['trade_date'].min())
        crisis_data = self._get_crisis_index(country)
        if crisis_data is not None and not crisis_data.empty and 'CX' in crisis_data.columns:
            crisis_sub = crisis_data[crisis_data.index >= start_d][['CX']].dropna()
            if not crisis_sub.empty:
                # CX ë³€???œì  ê¸°ì? ê·¸ë£¹??
                crisis_sub['cx_grp'] = (crisis_sub['CX'] != crisis_sub['CX'].shift()).cumsum()
                crisis_sub = crisis_sub.reset_index()
                crisis_sub.columns = ['date', 'CX', 'cx_grp']
                
                cx_periods = crisis_sub.groupby('cx_grp').agg(
                    start=('date', 'first'),
                    end=('date', 'last'),
                    cx_val=('CX', 'first')
                )
                cx_periods['next_date'] = cx_periods['end'].shift(-1).fillna(pd.Timestamp.now())
                
                # CX ê°’ì— ?°ë¥¸ ?‰ìƒ: 0=ì§„í•œ ë¹¨ê°•, 0.5=ì£¼í™©, 1=?¹ìƒ‰
                def cx_to_color(cx):
                    if pd.isna(cx):
                        return '#cccccc'
                    if cx <= 0:
                        return '#d62728'  # ì§„í•œ ë¹¨ê°•
                    elif cx < 0.5:
                        return '#ff6666'  # ?°í•œ ë¹¨ê°•
                    elif cx < 1.0:
                        return '#ffcc66'  # ì£¼í™©
                    else:
                        return '#2ca02c'  # ?¹ìƒ‰
                
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
        Crisis-Index (Disparity ê¸°ë°˜ buy/sell ? í˜¸) ì¡°íšŒ
        ?¸ë??ì„œ GD_utils ?¬ìš© ??override ê°€??
        """
        # ê¸°ë³¸: ìºì‹œ???°ì´??ë°˜í™˜ (?†ìœ¼ë©?None)
        if not hasattr(self, '_crisis_cache'):
            self._crisis_cache = {}
        return self._crisis_cache.get(country)
    
    def set_crisis_index(self, country: str, crisis_df: pd.DataFrame):
        """
        ?¸ë??ì„œ ê³„ì‚°??Crisis-Index ?¤ì •
        
        Args:
            country: êµ?? ì½”ë“œ
            crisis_df: 'cut' ì»¬ëŸ¼???ˆëŠ” DataFrame (index=date, cut='buy'/'sell')
        """
        if not hasattr(self, '_crisis_cache'):
            self._crisis_cache = {}
        self._crisis_cache[country] = crisis_df
    
    def _make_clocks(self, country):
        """Business Cycle Clock 3ê°??ì„±"""
        first_curve = self._first_curve.get(country)
        precomputed = self._precomputed_regimes.get(country)
        raw_data = self._raw_data.get(country)
        
        # Clock 1: First Value Only
        c1 = self._make_single_clock("1. First Value (Static)", 
                                     first_curve.tail(24) if first_curve is not None else None,
                                     compare=False)
        
        # Clock 2: PIT History (ê°?ë°œí‘œ ?œì ??ê¸°ë¡)
        if precomputed is not None and not precomputed.empty:
            pit_data = precomputed[['data_month', 'LEVEL', 'DIRECTION', 'exp2_regime', 'trade_date']].copy()
            pit_data = pit_data.rename(columns={'data_month': 'date', 'exp2_regime': 'ECI'})
            pit_data = pit_data.drop_duplicates(subset=['date'], keep='last')
            
            # First values ì¶”ê?
            pit_data['LEVEL_first'] = pit_data['date'].map(
                lambda d: self._first_vals_map.get(country, {}).get(d, {}).get('LEVEL', np.nan))
            pit_data['DIRECTION_first'] = pit_data['date'].map(
                lambda d: self._first_vals_map.get(country, {}).get(d, {}).get('DIRECTION', np.nan))
            pit_data['LEVEL'] = pit_data['LEVEL']
            pit_data['DIRECTION'] = pit_data['DIRECTION']
            
            c2 = self._make_single_clock("2. PIT History (Realized)", pit_data.tail(24), compare=True)
        else:
            c2 = self._make_single_clock("2. PIT History", None, compare=False)
        
        # Clock 3: Current Fresh Snapshot (?„ì¬ ?œì  ìµœì‹  ?°ì´??
        if raw_data is not None and not raw_data.empty:
            # ?„ì¬ ?œì ?ì„œ??fresh ECI ê³„ì‚°
            current_fresh = calculate_eci(raw_data[['date', 'value']].drop_duplicates(subset=['date'], keep='last'))
            if current_fresh is not None and not current_fresh.empty:
                current_fresh = current_fresh.tail(24).copy()
                # First values ì¶”ê?
                current_fresh['LEVEL_first'] = current_fresh['date'].map(
                    lambda d: self._first_vals_map.get(country, {}).get(d, {}).get('LEVEL', np.nan))
                current_fresh['DIRECTION_first'] = current_fresh['date'].map(
                    lambda d: self._first_vals_map.get(country, {}).get(d, {}).get('DIRECTION', np.nan))
                c3 = self._make_single_clock("3. Current Snapshot", current_fresh, compare=True)
            else:
                c3 = self._make_single_clock("3. Current Snapshot", None, compare=False)
        else:
            c3 = self._make_single_clock("3. Current Snapshot", None, compare=False)
        
        return c1, c2, c3
    
    def _make_single_clock(self, title, df, compare=False):
        """?¨ì¼ Business Cycle Clock ?ì„±"""
        from bokeh.plotting import figure
        from bokeh.models import ColumnDataSource, HoverTool, Range1d, BoxAnnotation, Arrow, NormalHead
        from bokeh.palettes import Blues256
        
        p = figure(title=title, width=380, height=380, tools="pan,wheel_zoom,reset")
        
        if df is None or df.empty or 'LEVEL' not in df.columns:
            axis_range = 4
            p.x_range = Range1d(-axis_range, axis_range)
            p.y_range = Range1d(-axis_range, axis_range)
            return p
        
        d = df.copy()
        
        # ì¶?ë²”ìœ„ ê³„ì‚°
        df_valid = d.dropna(subset=['LEVEL', 'DIRECTION'])
        if df_valid.empty:
            axis_range = 4
        else:
            x_vals = df_valid['LEVEL'].values
            y_vals = df_valid['DIRECTION'].values
            
            if compare and 'LEVEL_first' in df_valid.columns:
                x_first = df_valid['LEVEL_first'].dropna().values
                y_first = df_valid['DIRECTION_first'].dropna().values
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
        
        # ?¬ë¶„ë©?ë°°ê²½
        p.add_layout(BoxAnnotation(bottom=0, left=0, fill_color="#eaffea", fill_alpha=0.3))
        p.add_layout(BoxAnnotation(bottom=0, right=0, fill_color="#fff9c4", fill_alpha=0.3))
        p.add_layout(BoxAnnotation(top=0, right=0, fill_color="#ffe6e6", fill_alpha=0.3))
        p.add_layout(BoxAnnotation(top=0, left=0, fill_color="#fff3e0", fill_alpha=0.3))
        
        # ?¼ë²¨
        p.text([label_pos, label_pos, -label_pos, -label_pos],
               [label_pos, -label_pos, -label_pos, label_pos],
               text=["Expansion", "Slowdown", "Contraction", "Recovery"],
               text_color=["green", "orange", "red", "gold"], 
               text_align="center", text_font_style="bold", alpha=0.5)
        
        # ë¹„êµ??First ?¬ì¸??
        if compare and 'LEVEL_first' in d.columns:
            valid_first = d.dropna(subset=['LEVEL_first', 'DIRECTION_first'])
            if not valid_first.empty:
                p.circle(valid_first['LEVEL_first'], valid_first['DIRECTION_first'], 
                         color='gray', size=4, alpha=0.5, legend_label="First")
                
                # ?”ì‚´??
                d['dist'] = np.sqrt((d['LEVEL'] - d['LEVEL_first'])**2 + 
                                    (d['DIRECTION'] - d['DIRECTION_first'])**2)
                arrows = d[d['dist'] > 0.1].dropna(subset=['LEVEL', 'DIRECTION', 'LEVEL_first', 'DIRECTION_first'])
                if not arrows.empty:
                    src_arr = ColumnDataSource(arrows)
                    p.add_layout(Arrow(
                        end=NormalHead(size=5, fill_color="crimson", line_color="crimson"),
                        x_start='LEVEL_first', y_start='DIRECTION_first',
                        x_end='LEVEL', y_end='DIRECTION',
                        line_color="crimson", line_dash='dotted', source=src_arr
                    ))
        
        # ê²½ë¡œ ê·¸ë¦¬ê¸?
        x, y = d['LEVEL'].values, d['DIRECTION'].values
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[valid_mask], y[valid_mask]
        
        if len(x) > 1:
            cols = [Blues256[int(50 + i / (len(x) - 1) * 150)] for i in range(len(x) - 1)]
            p.segment(x[:-1], y[:-1], x[1:], y[1:], color=cols, line_width=3)
        
        # ?¬ì¸??
        if len(x) > 0:
            normal_d = d.dropna(subset=['LEVEL', 'DIRECTION'])
            src = ColumnDataSource(dict(
                x=normal_d['LEVEL'].values, 
                y=normal_d['DIRECTION'].values,
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
# ?¬ìš© ?ˆì‹œ ë°?ê²€ì¦?(Usage Examples & Validation)
# =============================================================================


if __name__ == '__main__':
    provider = RegimeProvider(use_cache=False)
    
    # Crisis-Index ê³„ì‚° (GD_utilsê°€ ?ˆì„ ?Œë§Œ)
    if gdu is not None:
        # USA??S&P500+NASDAQ ?‰ê· 
        USA1_df = gdu.get_data.disparity_df_v2('^GSPC')
        USA2_df = gdu.get_data.disparity_df_v2('^IXIC')
        USA_CX = USA1_df['CX'].add(USA2_df['CX']).div(2).dropna()
        
        # USA disparity??S&P500 ê¸°ì?
        USA_df = USA1_df.copy()
        USA_df['CX'] = USA_CX
        provider.set_crisis_index('USA', USA_df)
        
        # ?€êµ? USA_CX ?¬ìš©
        for country in provider.loaded_countries:
            if country == 'USA' or country == 'G7':
                continue
            ticker = INDEX_TICKER_MAP.get(country)
            if ticker:
                local_df = gdu.get_data.disparity_df_v2(ticker)
                local_df['CX'] = USA_CX
                provider.set_crisis_index(country, local_df)

        # G7?€ USA?€ ?™ì¼?˜ê²Œ
        if 'G7' in provider.loaded_countries:
            provider.set_crisis_index('G7', USA_df)
    
    # ?€?œë³´???ì„± (Crisis-Index ?¬í•¨)
    provider.generate_dashboard(open_browser=True)