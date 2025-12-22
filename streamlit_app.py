# -*- coding: utf-8 -*-
"""
Market Regime Monitoring Dashboard
Streamlit + Plotly 기반 웹 대시보드

GitHub 연동된 Streamlit Cloud에서 배포 가능
Main file path: streamlit_app.py (root)
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Plotly
import plotly.graph_objects as go

# 로컬 모듈 (패키지 경로 사용)
from MarketRegimeMonitoring.regime_provider import RegimeProvider, COUNTRY_MAP, REGIME_SETTINGS, INDEX_TICKER_MAP, calculate_eci
from utils.streamlit_utils import (
    disparity_df_v2,
    plot_cumulative_returns,
    plot_regime_strip,
    plot_business_clock,
    create_regime_summary_table,
    REGIME_COLORS
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
# Custom CSS
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .regime-expansion { background-color: #2ca02c; color: white; padding: 5px 10px; border-radius: 5px; }
    .regime-recovery { background-color: #ffce30; color: black; padding: 5px 10px; border-radius: 5px; }
    .regime-slowdown { background-color: #ff7f0e; color: white; padding: 5px 10px; border-radius: 5px; }
    .regime-contraction { background-color: #d62728; color: white; padding: 5px 10px; border-radius: 5px; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Sidebar
# =============================================================================
st.sidebar.title("⚙️ 설정")

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
use_cache = st.sidebar.checkbox("💾 캐시 사용", value=True, help="오늘 날짜 기준 캐시 사용 (API 호출 감소)")

# =============================================================================
# Data Loading (Cached)
# =============================================================================
@st.cache_resource(ttl=3600)  # 1시간 캐시
def load_provider(countries: tuple, use_cache: bool) -> RegimeProvider:
    """RegimeProvider 로딩 (캐싱)"""
    provider = RegimeProvider(countries=list(countries), use_cache=use_cache)
    return provider

@st.cache_data(ttl=3600)
def load_crisis_indices(countries: tuple) -> dict:
    """Crisis Index 계산 (캐싱)"""
    crisis_cache = {}
    
    # USA: S&P500 + NASDAQ 평균
    try:
        usa1_df = disparity_df_v2('^GSPC')
        usa2_df = disparity_df_v2('^IXIC')
        
        if not usa1_df.empty and not usa2_df.empty:
            usa_cx = usa1_df['CX'].add(usa2_df['CX'], fill_value=0).div(2).dropna()
            usa_df = usa1_df.copy()
            usa_df['CX'] = usa_cx
            crisis_cache['USA'] = usa_df
            
            # 타국: USA_CX 사용 (단순화)
            for country in countries:
                if country == 'USA' or country == 'G7':
                    continue
                ticker = INDEX_TICKER_MAP.get(country)
                if ticker:
                    try:
                        local_df = disparity_df_v2(ticker)
                        if not local_df.empty:
                            # USA CX와 평균 대신 USA_CX만 사용 (단순화)
                            local_df['CX'] = usa_cx.reindex(local_df.index).ffill()
                            crisis_cache[country] = local_df
                    except Exception as e:
                        st.warning(f"Crisis Index 로딩 실패 ({country}): {e}")
            
            # G7은 USA와 동일
            if 'G7' in countries:
                crisis_cache['G7'] = usa_df
                
    except Exception as e:
        st.warning(f"Crisis Index 로딩 실패: {e}")
    
    return crisis_cache

# 데이터 로딩
with st.spinner("📡 데이터 로딩 중..."):
    provider = load_provider(tuple(selected_countries), use_cache)
    crisis_indices = load_crisis_indices(tuple(selected_countries))
    
    # Crisis Index를 provider에 설정
    for country, crisis_df in crisis_indices.items():
        provider.set_crisis_index(country, crisis_df)
    
    # 가격 데이터 로딩
    prices = provider._load_price_data()

# =============================================================================
# Main Content
# =============================================================================
st.markdown('<div class="main-header">📊 Market Regime Dashboard</div>', unsafe_allow_html=True)
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
st.dataframe(styled_df, use_container_width=True, hide_index=True)

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
        crisis_data = crisis_indices.get(country)
        
        # 1. 누적 수익률 차트
        st.markdown("#### 📊 누적 수익률")
        fig_returns = plot_cumulative_returns(
            precomputed=precomputed,
            prices=prices,
            ticker=info['ticker'],
            country_name=info['name'],
            start_date=start_date,
            crisis_data=crisis_data
        )
        st.plotly_chart(fig_returns, use_container_width=True)
        
        # 2. 국면 스트립 차트
        st.markdown("#### 📅 국면 타임라인")
        fig_strip = plot_regime_strip(
            precomputed=precomputed,
            crisis_data=crisis_data,
            start_date=start_date
        )
        st.plotly_chart(fig_strip, use_container_width=True)
        
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
                st.plotly_chart(fig_c1, use_container_width=True)
            else:
                st.info("Clock 1: 데이터 없음")
        
        # Clock 2: PIT History
        with col2:
            if precomputed is not None and not precomputed.empty:
                pit_data = precomputed[['data_month', 'placement', 'velocity', 'exp2_regime', 'trade_date']].copy()
                pit_data = pit_data.rename(columns={'data_month': 'date', 'exp2_regime': 'ECI', 
                                                     'placement': 'PLACEMENT', 'velocity': 'VELOCITY'})
                pit_data = pit_data.drop_duplicates(subset=['date'], keep='last').tail(24)
                
                # First values 추가
                first_vals_map = provider._first_vals_map.get(country, {})
                pit_data['PLACEMENT_first'] = pit_data['date'].map(
                    lambda d: first_vals_map.get(d, {}).get('PLACEMENT', np.nan))
                pit_data['VELOCITY_first'] = pit_data['date'].map(
                    lambda d: first_vals_map.get(d, {}).get('VELOCITY', np.nan))
                
                fig_c2 = plot_business_clock(pit_data, "2. PIT History (Realized)", compare=True)
                st.plotly_chart(fig_c2, use_container_width=True)
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
                    current_fresh['PLACEMENT_first'] = current_fresh['date'].map(
                        lambda d: first_vals_map.get(d, {}).get('PLACEMENT', np.nan))
                    current_fresh['VELOCITY_first'] = current_fresh['date'].map(
                        lambda d: first_vals_map.get(d, {}).get('VELOCITY', np.nan))
                    
                    fig_c3 = plot_business_clock(current_fresh, "3. Current Snapshot", compare=True)
                    st.plotly_chart(fig_c3, use_container_width=True)
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
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)

# =============================================================================
# Footer
# =============================================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: 0.8rem;">
        📊 Market Regime Monitoring Dashboard | 
        Data Source: FRED (OECD CLI), Yahoo Finance | 
        Last Updated: {}
    </div>
    """.format(datetime.now().strftime('%Y-%m-%d %H:%M')),
    unsafe_allow_html=True
)
