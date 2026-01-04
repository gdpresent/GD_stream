# -*- coding: utf-8 -*-
"""
Market Regime Monitoring Dashboard
Streamlit + Plotly 기반 웹 대시보드

GitHub 연동된 Streamlit Cloud에서 배포 가능
Main file path: streamlit_app.py (root)
"""
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Streamlit Cloud 경로 설정 (로컬 패키지 인식용)
ROOT_DIR = Path(__file__).parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Plotly
import plotly.graph_objects as go

# 로컬 모듈 (패키지 경로 사용)
from MarketRegimeMonitoring.regime_provider import RegimeProvider, COUNTRY_MAP, REGIME_SETTINGS, INDEX_TICKER_MAP, calculate_eci
from utils.streamlit_utils import (
    plot_cumulative_returns,
    plot_regime_strip,
    plot_business_clock,
    create_regime_summary_table,
    REGIME_COLORS,
    get_fear_greed_data,
    get_vix_data,
    get_yield_spread,
    get_dxy_data,
    create_indicator_gauge,
    create_indicators_chart,
    get_index_returns,
    get_sector_returns,
    create_sector_heatmap,
    create_sector_timeseries,
    style_returns_dataframe,
    calculate_regime_statistics,
    create_regime_stats_chart,
    get_market_breadth,
    create_breadth_gauge
)

# Strategy 로직 직접 구현 (Streamlit Cloud 호환)
# v4: Binary Score + Top 2 + Inverse Volatility Tiebreak
SCORE_MAP_BINARY = {
    '팽창': 2, 
    '회복': 2, 
    '둔화': 0, 
    '침체': 0, 
    'Cash': -1, 
    'Half': -2, 
    'Skipped': -3
}

# Non-investable scores (명시적 마스킹)
NON_INVESTABLE_SCORES = [-1, -2, -3]  # Cash, Half, Skipped


def calc_strategy_weight(regime_df, univ, ticker_map, prices, top_n=2, min_score=0.5, vol_lookback=63):
    """
    v4 Binary + Inverse Volatility 전략
    
    핵심:
    1. Binary Score: CLI 방향만 봄 (상승=투자, 하락=미투자)
    2. Top 2 집중: 확신 높은 국가에 집중 투자
    3. Inverse Volatility Tiebreak: 동점 시 변동성 낮은 국가 우선
    
    성과: Sharpe 0.866 (lookahead bias 제거 후)
    """
    score_df = regime_df.replace(SCORE_MAP_BINARY)
    
    # 변동성 계산 (일별)
    vol_dict = {}
    for c in univ:
        ticker = ticker_map.get(c)
        if ticker and ticker in prices.columns:
            ret = prices[ticker].pct_change()
            vol = ret.rolling(vol_lookback).std() * np.sqrt(252)
            vol_dict[c] = vol
    vol_df = pd.DataFrame(vol_dict)
    
    weights = []
    for idx, row in score_df.iterrows():
        valid = {}
        
        for c in univ:
            score = row.get(c, 0)
            
            # 기본 필터: Non-investable 제외, min_score 초과
            if score in NON_INVESTABLE_SCORES or score <= min_score:
                continue
            
            # Inverse Volatility Tiebreak
            if idx in vol_df.index and c in vol_df.columns:
                vol = vol_df.loc[idx, c]
                if pd.isna(vol) or vol == 0:
                    vol = 0.2
            else:
                vol = 0.2
            
            # 변동성 낮을수록 점수 높음
            tiebreak_score = (0.3 - vol) * 2.0
            composite_score = score + tiebreak_score
            
            valid[c] = composite_score
        
        if not valid:
            # 투자 대상 없으면 100% 현금
            w_row = {c: 0.0 for c in univ}
            w_row['CASH'] = 1.0
        else:
            # Composite Score 순으로 Top N 선택
            sorted_list = sorted(valid.items(), key=lambda x: x[1], reverse=True)
            selected = [c for c, s in sorted_list[:top_n]]
            
            # 동일 비중 배분
            w_per_country = 1.0 / len(selected)
            w_row = {c: w_per_country if c in selected else 0.0 for c in univ}
            w_row['CASH'] = 0.0
        
        weights.append(w_row)
    
    return pd.DataFrame(weights, index=score_df.index)

# =============================================================================
# Page Config
# =============================================================================
st.set_page_config(
    page_title="Market Regime Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# =============================================================================
# Sidebar
# =============================================================================
st.sidebar.title("⚙️ Settings")

# 국가 선택
all_countries = list(COUNTRY_MAP.keys())
default_countries = ['USA', 'Korea', 'Japan', 'China', 'Germany', 'France', 'UK', 'India', 'Brazil']
selected_countries = st.sidebar.multiselect(
    "📍 국가 선택",
    options=all_countries,
    default=[c for c in default_countries if c in all_countries]
)

if not selected_countries:
    st.warning("최소 1개 이상의 국가를 선택해주세요.")
    st.stop()

# 캐시 옵션
use_cache = st.sidebar.checkbox("💾 캐시 사용", value=True, help="체크하면 캐시된 데이터 사용 (API 호출 감소)")

# 캐시 날짜 선택 (캐시 사용 시에만 표시)
selected_cache_date = None
if use_cache:
    import os
    cache_dir = os.path.join(os.path.dirname(__file__), 'MarketRegimeMonitoring', 'cache')
    
    available_dates = []
    if os.path.exists(cache_dir):
        files = os.listdir(cache_dir)
        dates_set = set()
        for f in files:
            if f.endswith('.parquet'):
                parts = f.replace('.parquet', '').split('_')
                if len(parts) >= 2:
                    date_str = parts[-1]
                    if len(date_str) == 10 and date_str[4] == '-' and date_str[7] == '-':
                        dates_set.add(date_str)
        available_dates = sorted(dates_set, reverse=True)
    
    if available_dates:
        cache_date_options = ["오늘 (최신 API)"] + available_dates
        cache_date_selection = st.sidebar.selectbox(
            "📅 캐시 날짜 선택",
            options=cache_date_options,
            index=0,
            help="과거 캐시 데이터로 전환 가능 (OECD revision 전 데이터 확인용)"
        )
        selected_cache_date = None if cache_date_selection == "오늘 (최신 API)" else cache_date_selection
    else:
        st.sidebar.info("저장된 캐시 없음 (첫 실행 시 자동 생성)")


# 새로고침 버튼
st.sidebar.markdown("---")
if st.sidebar.button("🔄 데이터 새로고침", help="캐시를 지우고 최신 데이터를 불러옵니다"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# =============================================================================
# Custom CSS
# =============================================================================
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    .regime-expansion { background-color: #2ca02c; color: white; padding: 5px 10px; border-radius: 5px; }
    .regime-recovery { background-color: #ffce30; color: black; padding: 5px 10px; border-radius: 5px; }
    .regime-slowdown { background-color: #ff7f0e; color: white; padding: 5px 10px; border-radius: 5px; }
    .regime-contraction { background-color: #d62728; color: white; padding: 5px 10px; border-radius: 5px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 50px; padding: 10px 20px; background-color: #f0f2f6; border-radius: 5px; }
    
    /* 반응형 모바일 */
    @media (max-width: 768px) {
        .main-header { font-size: 1.5rem; }
        .stTabs [data-baseweb="tab"] { padding: 5px 10px; font-size: 0.8rem; height: auto; }
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Data Loading (Cached)
# =============================================================================
@st.cache_resource(ttl=3600)  # 1시간 캐시
def load_provider_cached(countries: tuple, use_cache: bool, cache_date: str = None) -> RegimeProvider:
    """RegimeProvider 로딩 (캐싱) - 콜백 없이"""
    provider = RegimeProvider(countries=list(countries), use_cache=use_cache, cache_date=cache_date)
    return provider

def load_provider_with_progress(countries: tuple, use_cache: bool, cache_date: str,
                                  progress_bar, detail_text, start_time) -> RegimeProvider:
    """RegimeProvider 로딩 (진행 표시 포함)"""
    import time
    total = len(countries)
    
    def progress_callback(country: str, current: int, total: int, source: str):
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        
        source_emoji = "💾" if source == 'cache' else "🌐" if source == 'api' else "📍"
        detail_text.markdown(f"{source_emoji} `{country}` 로딩 중... ({current}/{total}) - ⏱️ {minutes}분 {seconds}초 경과")
        progress_bar.progress(int((current / total) * 50))
    
    provider = RegimeProvider(
        countries=list(countries), 
        use_cache=use_cache,
        cache_date=cache_date,
        progress_callback=progress_callback
    )
    return provider

# 데이터 로딩 (스플래시 스타일)
loading_container = st.empty()

# 캐시 체크: 이미 캐시된 경우 빠른 로딩
cache_key = f"provider_{hash(tuple(selected_countries))}_{use_cache}_{selected_cache_date}"
is_first_load = cache_key not in st.session_state

if is_first_load:
    with loading_container.container():
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem;">
            <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">📊</h1>
            <h2 style="color: #1f77b4; margin-bottom: 0.5rem;">Market Regime Dashboard</h2>
            <p style="color: #666; margin-bottom: 2rem;">데이터를 불러오는 중입니다...</p>
        </div>
        """, unsafe_allow_html=True)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        detail_text = st.empty()
        
        import time
        start_time = time.time()
        
        # Step 1: FRED 데이터 로딩 (콜백 포함)
        status_text.markdown("🌍 **FRED에서 데이터 로딩 중...**")
        provider = load_provider_with_progress(tuple(selected_countries), use_cache, selected_cache_date,
                                                progress_bar, detail_text, start_time)
        st.session_state[cache_key] = provider
        progress_bar.progress(50)
        
        # Step 2: 가격 데이터 로딩
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        
        status_text.markdown("💹 **Yahoo Finance에서 가격 데이터 로딩 중...**")
        detail_text.markdown(f"📊 주가 지수 데이터 수집 중... - ⏱️ {minutes}분 {seconds}초 경과")
        
        prices = provider._load_price_data()
        progress_bar.progress(100)
        
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        
        status_text.markdown(f"✅ **로딩 완료!** (총 {minutes}분 {seconds}초)")
        detail_text.empty()
    
    # 로딩 완료 후 로딩 화면 제거
    loading_container.empty()
else:
    # 캐시된 경우 빠른 로딩
    provider = st.session_state[cache_key]
    prices = provider._load_price_data()

# 로딩 화면 제거 (안전장치)
loading_container.empty()

# =============================================================================
# Main Content
# =============================================================================
st.markdown('<div class="main-header">📊 Market Regime Dashboard</div>', unsafe_allow_html=True)

# 캐시 날짜 표시
if selected_cache_date:
    st.info(f"📅 **캐시 데이터 기준일: {selected_cache_date}** (과거 시점 데이터)")
else:
    today_str = datetime.now().strftime('%Y-%m-%d')
    st.caption(f"📅 데이터 기준일: {today_str} (최신)")

st.markdown("---")

# =============================================================================
# Market Indicators 데이터 로딩 (Fear & Greed, Yield Spread, DXY)
# =============================================================================
@st.cache_data(ttl=3600)
def load_market_indicators():
    fear_greed = get_fear_greed_data()
    spread = get_yield_spread()
    dxy = get_dxy_data()
    vix = get_vix_data()  # 시계열 차트용
    return fear_greed, spread, dxy, vix

@st.cache_data(ttl=3600)
def load_market_breadth():
    return get_market_breadth()

fear_greed_data, spread_df, dxy_df, vix_df = load_market_indicators()
breadth_data = load_market_breadth()

# 현재 값 추출
fg_current = fear_greed_data.get('current')
fg_text = fear_greed_data.get('current_text', '')
spread_current = spread_df['Spread'].iloc[-1] if not spread_df.empty else None
dxy_current = dxy_df['DXY'].iloc[-1] if not dxy_df.empty else None
breadth_position = breadth_data.get('position')
breadth_status = breadth_data.get('status', 'N/A')

# =============================================================================
# Global Index Returns (Phase 2)
# =============================================================================
st.subheader("📈 글로벌 지수 현황")

@st.cache_data(ttl=3600)
def load_index_returns():
    return get_index_returns()  # returns (df, reference_date)

@st.cache_data(ttl=3600)
def load_sector_returns(reference_date=None):
    return get_sector_returns(reference_date)

# 탭으로 구분
idx_tab1, idx_tab2 = st.tabs(["🌐 글로벌 지수", "🏭 섹터별 현황"])

# 기준일자 로딩
index_result = load_index_returns()
if isinstance(index_result, tuple):
    index_df, reference_date = index_result
else:
    index_df = index_result
    reference_date = pd.Timestamp.now().normalize()

ref_date_str = reference_date.strftime('%Y-%m-%d') if reference_date else ''

with idx_tab1:
    if not index_df.empty:
        styled_index = style_returns_dataframe(index_df)
        st.dataframe(styled_index, use_container_width=True, hide_index=True)
        st.caption(f"📅 기준일: {ref_date_str} (미국 시장 종가 기준)")
    else:
        st.info("지수 데이터를 불러오는 중...")

with idx_tab2:
    sector_df = load_sector_returns(reference_date)  # 동일한 기준일 사용
    if not sector_df.empty:
        # 히트맵 표시
        fig_heatmap = create_sector_heatmap(sector_df)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        st.caption(f"📅 기준일: {ref_date_str}")
        
        # 시계열 차트 (접기)
        with st.expander("📈 섹터 ETF 시계열"):
            sector_days = st.radio("기간 선택", [90, 180, 365], index=1, horizontal=True, format_func=lambda x: f"{x}일")
            fig_sector_ts = create_sector_timeseries(days=sector_days)
            st.plotly_chart(fig_sector_ts, use_container_width=True)
    else:
        st.info("섹터 데이터를 불러오는 중...")

st.markdown("---")

# 현재 국면 요약 테이블
st.subheader("🌍 현재 국면 요약")
summary_df = create_regime_summary_table(provider, selected_countries)

# 색상 적용 함수
def color_regime(val):
    colors = {
        '팽창': 'background-color: #2ca02c; color: white',
        '회복': 'background-color: #ffce30; color: black',
        '둔화': 'background-color: #ff7f0e; color: white',
        '침체': 'background-color: #d62728; color: white',
        'Cash': 'background-color: #ffb347; color: black',
        'Half': 'background-color: #9467bd; color: white',
        'Skipped': 'background-color: #f0f0f0; color: black'
    }
    return colors.get(val, '')

styled_df = summary_df.style.applymap(
    color_regime, 
    subset=['First', 'Fresh', 'Smart']
)
st.dataframe(styled_df, use_container_width=True, hide_index=True)

st.markdown("---")

# =============================================================================
# Rotation Strategy Section
# =============================================================================
st.subheader("🎯 ETF Rotation Strategy")
st.caption("📊 v4: BiInvVol | Top 2 집중 | 변동성 낮은 국가 우선")

# Strategy 유니버스 (성과 나쁜 순 - lookahead bias 제거)
Univ = ['Brazil', 'China', 'Japan', 'UK', 'France', 'India', 'Germany', 'Korea', 'USA']
ticker_map = {c: COUNTRY_MAP[c]['ticker'] for c in Univ if c in COUNTRY_MAP}

# v4 파라미터
top_n = 2              # Top 2 집중
min_score = 0.5        # Binary에서 score > 0.5 (팽창/회복만)
ensemble_method = 'first'  # First Value 기준

# 가격 데이터
prices = provider._load_price_data()

# Regime 데이터 수집 (이미 로드된 provider 사용)
try:
    regime_col = {'first': 'exp1_regime', 'fresh': 'exp2_regime', 'smart': 'exp3_regime'}[ensemble_method]
    
    # 각 국가의 regime 데이터를 합침
    regime_data = {}
    for country in Univ:
        precomputed = provider._precomputed_regimes.get(country)
        if precomputed is not None and not precomputed.empty:
            sub = precomputed[['trade_date', regime_col]].copy()
            sub = sub.set_index('trade_date')
            regime_data[country] = sub[regime_col]
    
    if regime_data:
        regime_df = pd.DataFrame(regime_data)
        regime_df = regime_df.ffill().dropna(how='all')
        score_df = regime_df.replace(SCORE_MAP_BINARY)
        
        # Weight 계산 (v4 Binary + InvVol)
        w = calc_strategy_weight(regime_df, Univ, ticker_map, prices, top_n, min_score)
    
        # 현재 포지션 표시
        if not w.empty:
            st.markdown("#### 📍 현재 포지션")
        
            latest_w = w.iloc[-1]
            latest_regime = regime_df.iloc[-1]
            latest_score = score_df.iloc[-1]
            latest_date = w.index[-1]
        
            # 투자 중인 국가만 필터
            investing = [(c, latest_w[c], latest_regime[c], latest_score[c]) 
                         for c in Univ if c in latest_w.index and latest_w[c] > 0.001]
        
            if investing:
                pos_data = []
                for country, weight, regime, score in investing:
                    pos_data.append({
                        '국가': country,
                        'Ticker': ticker_map.get(country, '-'),
                        'Regime': regime,
                        'Score': int(score),
                        '비중': f"{weight:.1%}"
                    })
            
                pos_df = pd.DataFrame(pos_data)
            
                # Regime 색상 적용
                def color_regime_pos(val):
                    colors = {
                        '팽창': 'background-color: #2ca02c; color: white',
                        '회복': 'background-color: #ffce30; color: black',
                        '둔화': 'background-color: #ff7f0e; color: white',
                        '침체': 'background-color: #d62728; color: white',
                    }
                    return colors.get(val, '')
            
                styled_pos = pos_df.style.applymap(color_regime_pos, subset=['Regime'])
            
                pos_cols = st.columns([3, 1])
                with pos_cols[0]:
                    st.dataframe(styled_pos, hide_index=True, use_container_width=True)
                with pos_cols[1]:
                    cash_pct = latest_w.get('CASH', 0)
                    st.metric("현금 비중", f"{cash_pct:.1%}")
                    st.caption(f"마지막 업데이트: {latest_date.strftime('%Y-%m-%d')}")
            else:
                st.warning("추천 포지션 없음 (전액 CASH)")
        
            # 포트폴리오 Pie Chart
            with st.expander("🥧 포트폴리오 구성"):
                pie_data = [(c, latest_w[c]) for c in Univ + ['CASH'] 
                            if c in latest_w.index and latest_w[c] > 0.001]
                if pie_data:
                    labels = [d[0] for d in pie_data]
                    values = [d[1] for d in pie_data]
                    pie_colors = ['#2ca02c' if v > 0.2 else '#1f77b4' for v in values]
                    pie_colors[-1] = '#cccccc' if labels[-1] == 'CASH' else pie_colors[-1]
                
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=labels, values=values,
                        hole=0.4,
                        marker_colors=pie_colors,
                        textinfo='label+percent',
                        hovertemplate='%{label}: %{percent}<extra></extra>'
                    )])
                    fig_pie.update_layout(height=300, margin=dict(t=20, b=20, l=20, r=20))
                    st.plotly_chart(fig_pie, use_container_width=True)
        
            # 최근 리밸런싱 이력
            with st.expander("📅 리밸런싱 이력 (최근 10회)"):
                w_display = (w[Univ + ['CASH']] * 100).round(1).tail(10)
                w_display.index = w_display.index.strftime('%Y-%m-%d')
                st.dataframe(w_display, use_container_width=True)
            
                # 회전율 계산
                turnover = (w.diff().abs().sum(axis=1) / 2).mean()
                st.caption(f"평균 회전율: {turnover:.1%} / 리밸런싱")
        
            # 누적수익률 차트
            st.markdown("#### 📈 누적 수익률 (Backtest)")
        
            # 백테스트 수익률 계산
            try:
                # 가격 데이터를 prices에서 가져옴 (이미 로딩됨)
                w_ticker = w.rename(columns=lambda x: ticker_map.get(x, x))
            
                # 백테스트 시작일 = weight 데이터 시작일
                backtest_start = w_ticker.index[0]
            
                # 일별 수익률 계산 (백테스트 시작일 이후만)
                daily_ret = prices.pct_change().fillna(0)
                daily_ret = daily_ret.loc[daily_ret.index >= backtest_start]
            
                # 전략 수익률 계산
                common_idx = w_ticker.index.intersection(daily_ret.index)
                if len(common_idx) > 0:
                    # Forward fill weights to daily
                    w_daily = w_ticker.reindex(daily_ret.index).ffill()
                    w_daily = w_daily.loc[w_daily.index >= backtest_start]
                
                    # NaN 제거 (첫 날 이전 데이터)
                    w_daily = w_daily.dropna(how='all')
                
                    # 포트폴리오 일별 수익률
                    port_ret = (w_daily.shift(1) * daily_ret.reindex(columns=w_daily.columns, fill_value=0)).sum(axis=1)
                    port_ret = port_ret.dropna()
                    port_ret = port_ret.loc[port_ret.index >= backtest_start]
                
                    # 누적 수익률
                    strat_cum = (1 + port_ret).cumprod()
                
                    # ACWI 벤치마크 (같은 시작일)
                    if 'ACWI' in prices.columns:
                        bm_ret = daily_ret['ACWI']
                    else:
                        # ACWI 없으면 Equal Weight fallback
                        ew_tickers = [ticker_map.get(c) for c in Univ if c in ticker_map]
                        bm_ret = daily_ret[ew_tickers].mean(axis=1)
                    bm_ret = bm_ret.loc[strat_cum.index]
                    bm_cum = (1 + bm_ret).cumprod()
                
                    # Plotly 차트
                    fig_cum = go.Figure()
                
                    fig_cum.add_trace(go.Scatter(
                        x=strat_cum.index, y=strat_cum.values,
                        name=f'Strategy ({ensemble_method.upper()})',
                        line=dict(color='#2ca02c', width=2),
                        hovertemplate='%{x|%Y-%m-%d}<br>Return: %{y:.1%}<extra></extra>'
                    ))
                
                    fig_cum.add_trace(go.Scatter(
                        x=bm_cum.index, y=bm_cum.values,
                        name='ACWI (BM)',
                        line=dict(color='silver', width=2, dash='dash'),
                        hovertemplate='%{x|%Y-%m-%d}<br>Return: %{y:.1%}<extra></extra>'
                    ))
                
                    fig_cum.update_layout(
                        height=350,
                        xaxis_title='Date',
                        yaxis_title='Cumulative Return',
                        yaxis_tickformat='.0%',
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                        hovermode='x unified',
                        margin=dict(t=30, b=30, l=50, r=20)
                    )
                
                    st.plotly_chart(fig_cum, use_container_width=True)
                
                    # 성과 지표
                    if len(strat_cum) > 252:
                        yrs = (strat_cum.index[-1] - strat_cum.index[0]).days / 365.25
                        total_ret = strat_cum.iloc[-1] - 1
                        cagr = (1 + total_ret) ** (1/yrs) - 1 if yrs > 0 else 0
                        vol = port_ret.std() * np.sqrt(252)
                        sharpe = (cagr - 0.02) / vol if vol > 0 else 0
                    
                        # MDD
                        rolling_max = strat_cum.expanding().max()
                        drawdown = (strat_cum - rolling_max) / rolling_max
                        mdd = drawdown.min()
                    
                        # ACWI 성과
                        bm_total = bm_cum.iloc[-1] - 1
                        bm_cagr = (1 + bm_total) ** (1/yrs) - 1 if yrs > 0 else 0
                        bm_vol = bm_ret.std() * np.sqrt(252)
                        bm_sharpe = (bm_cagr - 0.02) / bm_vol if bm_vol > 0 else 0
                        bm_rm = bm_cum.expanding().max()
                        bm_mdd = ((bm_cum - bm_rm) / bm_rm).min()
                    
                        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
                        with perf_col1:
                            st.metric("CAGR", f"{cagr:.1%}", delta=f"{(cagr - bm_cagr)*100:.1f}%p vs BM")
                        with perf_col2:
                            st.metric("Sharpe", f"{sharpe:.2f}", delta=f"{sharpe - bm_sharpe:+.2f} vs BM")
                        with perf_col3:
                            st.metric("MDD", f"{mdd:.1%}", delta=f"{(mdd - bm_mdd)*100:.1f}%p" if mdd > bm_mdd else f"{(mdd - bm_mdd)*100:+.1f}%p")
                        with perf_col4:
                            st.metric("Vol", f"{vol:.1%}")
                else:
                    st.info("가격 데이터와 매칭되는 기간이 없습니다.")
                
            except Exception as e:
                st.warning(f"누적수익률 차트 생성 중 오류: {e}")

except Exception as e:
    st.error(f"Strategy 계산 오류: {e}")
    import traceback
    st.code(traceback.format_exc())

st.markdown("---")

# 국가별 상세 차트
st.subheader("📈 국가별 상세 분석")

# 탭으로 국가 선택
tabs = st.tabs([f"🏳️ {COUNTRY_MAP[c]['name']}" for c in selected_countries])

for i, country in enumerate(selected_countries):
    with tabs[i]:
        info = COUNTRY_MAP[country]
        precomputed = provider._precomputed_regimes.get(country)
        
        if precomputed is None or precomputed.empty:
            st.warning(f"{country}: 데이터 없음")
            continue
        
        start_date = provider._effective_start.get(country, precomputed['trade_date'].min())
        
        # 1. 누적 수익률 차트
        st.markdown("#### 📊 누적 수익률")
        fig_returns = plot_cumulative_returns(
            precomputed=precomputed,
            prices=prices,
            ticker=info['ticker'],
            country_name=info['name'],
            start_date=start_date,
            crisis_data=None
        )
        st.plotly_chart(fig_returns, use_container_width=True)
        
        # 2. 국면 스트립 차트
        st.markdown("#### 📅 국면 타임라인")
        fig_strip = plot_regime_strip(
            precomputed=precomputed,
            crisis_data=None,
            start_date=start_date
        )
        st.plotly_chart(fig_strip, use_container_width=True)
        
        # 2.5. 국면별 통계 (Phase 2)
        with st.expander("📊 국면별 수익률 통계"):
            regime_stats = calculate_regime_statistics(
                precomputed=precomputed,
                prices=prices,
                ticker=info['ticker'],
                method='fresh'
            )
            if not regime_stats.empty:
                # 바 차트
                fig_stats = create_regime_stats_chart(regime_stats)
                st.plotly_chart(fig_stats, use_container_width=True)
                
                # 통계 테이블
                st.dataframe(
                    regime_stats.style.format({
                        '평균 일수익률': '{:.3f}%',
                        '연환산 수익률': '{:.1f}%',
                        '변동성(연환산)': '{:.1f}%',
                        '최대 수익': '{:.2f}%',
                        '최대 손실': '{:.2f}%',
                        '샤프비율': '{:.2f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("통계 데이터 없음")
        
        # 3. Business Cycle Clocks (Plotly 개선 버전)
        st.markdown("#### 🕐 Business Cycle Clock")
        
        col1, col2, col3 = st.columns(3)
        
        # Clock 1: First Value
        first_curve = provider._first_curve.get(country)
        with col1:
            if first_curve is not None and not first_curve.empty:
                fig_c1 = plot_business_clock(
                    first_curve.tail(24).copy(),
                    "1. First Value (Static)",
                    compare=False
                )
                st.plotly_chart(fig_c1, use_container_width=True, config={'displayModeBar': False})
            else:
                st.info("Clock 1: 데이터 없음")
        
        # Clock 2: PIT History
        with col2:
            if precomputed is not None and not precomputed.empty:
                pit_data = precomputed[['data_month', 'Level', 'Momentum', 'exp2_regime', 'trade_date']].copy()
                pit_data = pit_data.rename(columns={'data_month': 'date', 'exp2_regime': 'Regime', 
                                                     'Level': 'Level', 'Momentum': 'Momentum'})
                pit_data = pit_data.drop_duplicates(subset=['date'], keep='last').tail(24)
                
                # First values 추가
                first_vals_map = provider._first_vals_map.get(country, {})
                pit_data['Level_first'] = pit_data['date'].map(lambda d: first_vals_map.get(d, {}).get('Level', np.nan))
                pit_data['Momentum_first'] = pit_data['date'].map(lambda d: first_vals_map.get(d, {}).get('Momentum', np.nan))
                
                fig_c2 = plot_business_clock(pit_data, "2. PIT History (Realized)", compare=True)
                st.plotly_chart(fig_c2, use_container_width=True, config={'displayModeBar': False})
            else:
                st.info("Clock 2: 데이터 없음")
        
        # Clock 3: Current Snapshot
        with col3:
            raw_data = provider._raw_data.get(country)
            if raw_data is not None and not raw_data.empty:
                current_fresh = calculate_eci(
                    raw_data[['date', 'value']].drop_duplicates(subset=['date'], keep='last')
                )
                if current_fresh is not None and not current_fresh.empty:
                    current_fresh = current_fresh.tail(24).copy()
                    
                    first_vals_map = provider._first_vals_map.get(country, {})
                    current_fresh['Level_first'] = current_fresh['date'].map(lambda d: first_vals_map.get(d, {}).get('Level', np.nan))
                    current_fresh['Momentum_first'] = current_fresh['date'].map(lambda d: first_vals_map.get(d, {}).get('Momentum', np.nan))
                    
                    fig_c3 = plot_business_clock(current_fresh, "3. Current Snapshot", compare=True)
                    st.plotly_chart(fig_c3, use_container_width=True, config={'displayModeBar': False})
                else:
                    st.info("Clock 3: 데이터 없음")
            else:
                st.info("Clock 3: 데이터 없음")
        
        # 4. 상세 데이터 테이블 (접기)
        with st.expander("📋 상세 데이터 보기"):
            # 발표일 정보 포함 컬럼 선택
            cols_to_show = ['trade_date', 'realtime_start', 'data_month', 
                           'first_regime', 'fresh_regime', 'smart_regime',
                           'expected_next_data', 'expected_next_release', 'is_missing']
            
            available_cols = [c for c in cols_to_show if c in precomputed.columns]
            display_df = precomputed[available_cols].tail(24).copy()
            
            # 날짜 포맷팅
            if 'trade_date' in display_df.columns:
                display_df['trade_date'] = display_df['trade_date'].dt.strftime('%Y-%m-%d')
            if 'realtime_start' in display_df.columns:
                display_df['realtime_start'] = display_df['realtime_start'].dt.strftime('%Y-%m-%d')
            if 'data_month' in display_df.columns:
                display_df['data_month'] = display_df['data_month'].dt.strftime('%Y-%m')
            if 'expected_next_data' in display_df.columns:
                display_df['expected_next_data'] = pd.to_datetime(display_df['expected_next_data']).dt.strftime('%Y-%m')
            if 'expected_next_release' in display_df.columns:
                display_df['expected_next_release'] = pd.to_datetime(display_df['expected_next_release']).dt.strftime('%Y-%m-%d')
            
            # 컬럼명 한글화
            col_rename = {
                'trade_date': '거래일',
                'realtime_start': '발표일',
                'data_month': '데이터월',
                'first_regime': 'First',
                'fresh_regime': 'Fresh',
                'smart_regime': 'Smart',
                'expected_next_data': '다음예상데이터',
                'expected_next_release': '다음예상발표일',
                'is_missing': 'Skipped'
            }
            display_df = display_df.rename(columns={k: v for k, v in col_rename.items() if k in display_df.columns})
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)

# =============================================================================
# Fear & Greed (맨 아래 작게 배치)
# =============================================================================
st.markdown("---")
st.subheader("💭 시장 심리")

fg_col1, fg_col2, fg_col3 = st.columns([1, 2, 1])
with fg_col2:
    if fg_current is not None:
        fig_fg = create_indicator_gauge(
            fg_current, "CNN Fear & Greed Index", 0, 100,
            thresholds={'low': 25, 'high': 75},
            reverse_colors=True
        )
        st.plotly_chart(fig_fg, use_container_width=True, config={'displayModeBar': False})
        if fg_text:
            st.markdown(f"<center><b>{fg_text}</b></center>", unsafe_allow_html=True)
    else:
        st.info("Fear & Greed 데이터 없음")

# =============================================================================
# Footer
# =============================================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: 0.8rem;">
        📊 Market Regime Monitoring Dashboard | 
        Data Source: FRED, Yahoo Finance | 
        Last Updated: {}
    </div>
    """.format(datetime.now().strftime('%Y-%m-%d %H:%M')),
    unsafe_allow_html=True
)
