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
use_cache = st.sidebar.checkbox("💾 캐시 사용", value=False, help="체크하면 오늘 날짜 기준 캐시 사용 (API 호출 감소)")

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
def load_provider(countries: tuple, use_cache: bool) -> RegimeProvider:
    """RegimeProvider 로딩 (캐싱)"""
    provider = RegimeProvider(countries=list(countries), use_cache=use_cache)
    return provider

# 데이터 로딩 (스플래시 스타일)
loading_container = st.empty()

with loading_container.container():
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem;">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">�</h1>
        <h2 style="color: #1f77b4; margin-bottom: 0.5rem;">Market Regime Dashboard</h2>
        <p style="color: #666; margin-bottom: 2rem;">데이터를 불러오는 중입니다...</p>
    </div>
    """, unsafe_allow_html=True)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: CLI 데이터 로딩
    status_text.markdown("🌍 **데이터 로딩 중...**")
    provider = load_provider(tuple(selected_countries), use_cache)
    progress_bar.progress(50)
    
    # Step 2: 가격 데이터 로딩
    status_text.markdown("💹 **가격 데이터 로딩 중...**")
    prices = provider._load_price_data()
    progress_bar.progress(100)
    status_text.markdown("✅ **로딩 완료!**")

# 로딩 완료 후 로딩 화면 제거
loading_container.empty()

# =============================================================================
# Main Content
# =============================================================================
st.markdown('<div class="main-header">📊 Market Regime Dashboard</div>', unsafe_allow_html=True)
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
        st.dataframe(styled_index, width='stretch', hide_index=True)
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
    subset=['Exp1 (First)', 'Exp2 (Fresh)', 'Exp3 (Smart)']
)
st.dataframe(styled_df, width='stretch', hide_index=True)

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
                method='exp2'
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
                pit_data = pit_data.rename(columns={'data_month': 'date', 'exp2_regime': 'ECI', 
                                                     'Level': 'Level', 'Momentum': 'Momentum'})
                pit_data = pit_data.drop_duplicates(subset=['date'], keep='last').tail(24)
                
                # First values 추가
                first_vals_map = provider._first_vals_map.get(country, {})
                pit_data['Level_first'] = pit_data['date'].map(
                    lambda d: first_vals_map.get(d, {}).get('Level', np.nan))
                pit_data['Momentum_first'] = pit_data['date'].map(
                    lambda d: first_vals_map.get(d, {}).get('Momentum', np.nan))
                
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
                    current_fresh['Level_first'] = current_fresh['date'].map(
                        lambda d: first_vals_map.get(d, {}).get('Level', np.nan))
                    current_fresh['Momentum_first'] = current_fresh['date'].map(
                        lambda d: first_vals_map.get(d, {}).get('Momentum', np.nan))
                    
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
                           'exp1_regime', 'exp2_regime', 'exp3_regime',
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
                'exp1_regime': 'Exp1',
                'exp2_regime': 'Exp2',
                'exp3_regime': 'Exp3',
                'expected_next_data': '다음예상데이터',
                'expected_next_release': '다음예상발표일',
                'is_missing': 'Skipped'
            }
            display_df = display_df.rename(columns={k: v for k, v in col_rename.items() if k in display_df.columns})
            
            st.dataframe(display_df, width='stretch', hide_index=True)

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
