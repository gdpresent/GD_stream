# -*- coding: utf-8 -*-
"""
Market Regime Monitoring Dashboard
Streamlit + Plotly ê¸°ë°˜ ???€?œë³´??

GitHub ?°ë™??Streamlit Cloud?ì„œ ë°°í¬ ê°€??
Main file path: streamlit_app.py (root)
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Streamlit Cloud ê²½ë¡œ ?¤ì • (ë¡œì»¬ ?¨í‚¤ì§€ ?¸ì‹??
ROOT_DIR = Path(__file__).parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Plotly
import plotly.graph_objects as go

# ë¡œì»¬ ëª¨ë“ˆ (?¨í‚¤ì§€ ê²½ë¡œ ?¬ìš©)
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
    page_icon="?“Š",
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
st.sidebar.title("?™ï¸ ?¤ì •")

# êµ?? ? íƒ
all_countries = list(COUNTRY_MAP.keys())
default_countries = ['USA', 'Korea', 'Japan', 'China', 'Germany', 'France', 'UK', 'India', 'Brazil']
selected_countries = st.sidebar.multiselect(
    "?“ êµ?? ? íƒ",
    options=all_countries,
    default=[c for c in default_countries if c in all_countries]
)

if not selected_countries:
    st.warning("ìµœì†Œ 1ê°??´ìƒ??êµ??ë¥?? íƒ?´ì£¼?¸ìš”.")
    st.stop()

# ìºì‹œ ?µì…˜
use_cache = st.sidebar.checkbox("?’¾ ìºì‹œ ?¬ìš©", value=False, help="ì²´í¬?˜ë©´ ?¤ëŠ˜ ? ì§œ ê¸°ì? ìºì‹œ ?¬ìš© (API ?¸ì¶œ ê°ì†Œ)")

# =============================================================================
# Data Loading (Cached)
# =============================================================================
@st.cache_resource(ttl=3600)  # 1?œê°„ ìºì‹œ
def load_provider(countries: tuple, use_cache: bool) -> RegimeProvider:
    """RegimeProvider ë¡œë”© (ìºì‹±)"""
    provider = RegimeProvider(countries=list(countries), use_cache=use_cache)
    return provider

@st.cache_data(ttl=3600)
def load_crisis_indices(countries: tuple) -> dict:
    """Crisis Index ê³„ì‚° (ìºì‹±)"""
    crisis_cache = {}
    
    # USA: S&P500 + NASDAQ ?‰ê· 
    try:
        usa1_df = disparity_df_v2('^GSPC')
        usa2_df = disparity_df_v2('^IXIC')
        
        if not usa1_df.empty and not usa2_df.empty:
            usa_cx = usa1_df['CX'].add(usa2_df['CX'], fill_value=0).div(2).dropna()
            usa_df = usa1_df.copy()
            usa_df['CX'] = usa_cx
            crisis_cache['USA'] = usa_df
            
            # ?€êµ? USA_CX ?¬ìš© (?¨ìˆœ??
            for country in countries:
                if country == 'USA' or country == 'G7':
                    continue
                ticker = INDEX_TICKER_MAP.get(country)
                if ticker:
                    try:
                        local_df = disparity_df_v2(ticker)
                        if not local_df.empty:
                            # USA CX?€ ?‰ê·  ?€??USA_CXë§??¬ìš© (?¨ìˆœ??
                            local_df['CX'] = usa_cx.reindex(local_df.index).ffill()
                            crisis_cache[country] = local_df
                    except Exception as e:
                        st.warning(f"Crisis Index ë¡œë”© ?¤íŒ¨ ({country}): {e}")
            
            # G7?€ USA?€ ?™ì¼
            if 'G7' in countries:
                crisis_cache['G7'] = usa_df
                
    except Exception as e:
        st.warning(f"Crisis Index ë¡œë”© ?¤íŒ¨: {e}")
    
    return crisis_cache

# ?°ì´??ë¡œë”© (?¤í”Œ?˜ì‹œ ?¤í???
loading_container = st.empty()

with loading_container.container():
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem;">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">ï¿?/h1>
        <h2 style="color: #1f77b4; margin-bottom: 0.5rem;">Market Regime Dashboard</h2>
        <p style="color: #666; margin-bottom: 2rem;">?°ì´?°ë? ë¶ˆëŸ¬?¤ëŠ” ì¤‘ì…?ˆë‹¤...</p>
    </div>
    """, unsafe_allow_html=True)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: CLI ?°ì´??ë¡œë”©
    status_text.markdown("?Œ **?°ì´??ë¡œë”© ì¤?..**")
    provider = load_provider(tuple(selected_countries), use_cache)
    progress_bar.progress(40)
    
    # Step 2: Crisis Index ê³„ì‚°
    status_text.markdown("?“ˆ **Crisis Index ê³„ì‚° ì¤?..**")
    crisis_indices = load_crisis_indices(tuple(selected_countries))
    progress_bar.progress(60)
    
    # Step 3: ?°ì´???°ê²°
    status_text.markdown("?”— **?°ì´???°ê²° ì¤?..**")
    for country, crisis_df in crisis_indices.items():
        provider.set_crisis_index(country, crisis_df)
    progress_bar.progress(80)
    
    # Step 4: ê°€ê²??°ì´??ë¡œë”©
    status_text.markdown("?’¹ **ê°€ê²??°ì´??ë¡œë”© ì¤?..**")
    prices = provider._load_price_data()
    progress_bar.progress(100)
    status_text.markdown("??**ë¡œë”© ?„ë£Œ!**")

# ë¡œë”© ?„ë£Œ ??ë¡œë”© ?”ë©´ ?œê±°
loading_container.empty()

# =============================================================================
# Main Content
# =============================================================================
st.markdown('<div class="main-header">?“Š Market Regime Dashboard</div>', unsafe_allow_html=True)
st.markdown("---")

# ?„ì¬ êµ?©´ ?”ì•½ ?Œì´ë¸?
st.subheader("?Œ ?„ì¬ êµ?©´ ?”ì•½")
summary_df = create_regime_summary_table(provider, selected_countries)

# ?‰ìƒ ?ìš© ?¨ìˆ˜
def color_regime(val):
    colors = {
        '?½ì°½': 'background-color: #2ca02c; color: white',
        '?Œë³µ': 'background-color: #ffce30; color: black',
        '?”í™”': 'background-color: #ff7f0e; color: white',
        'ì¹¨ì²´': 'background-color: #d62728; color: white',
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

# êµ??ë³??ì„¸ ì°¨íŠ¸
st.subheader("?“ˆ êµ??ë³??ì„¸ ë¶„ì„")

# ??œ¼ë¡?êµ?? ? íƒ
tabs = st.tabs([f"?³ï¸?{COUNTRY_MAP[c]['name']}" for c in selected_countries])

for i, country in enumerate(selected_countries):
    with tabs[i]:
        info = COUNTRY_MAP[country]
        precomputed = provider._precomputed_regimes.get(country)
        
        if precomputed is None or precomputed.empty:
            st.warning(f"{country}: ?°ì´???†ìŒ")
            continue
        
        start_date = provider._effective_start.get(country, precomputed['trade_date'].min())
        crisis_data = crisis_indices.get(country)
        
        # 1. ?„ì  ?˜ìµë¥?ì°¨íŠ¸
        st.markdown("#### ?“Š ?„ì  ?˜ìµë¥?)
        fig_returns = plot_cumulative_returns(
            precomputed=precomputed,
            prices=prices,
            ticker=info['ticker'],
            country_name=info['name'],
            start_date=start_date,
            crisis_data=crisis_data
        )
        st.plotly_chart(fig_returns, use_container_width=True)
        
        # 2. êµ?©´ ?¤íŠ¸ë¦?ì°¨íŠ¸
        st.markdown("#### ?“… êµ?©´ ?€?„ë¼??)
        fig_strip = plot_regime_strip(
            precomputed=precomputed,
            crisis_data=crisis_data,
            start_date=start_date
        )
        st.plotly_chart(fig_strip, use_container_width=True)
        
        # 3. Business Cycle Clocks (Plotly ê°œì„  ë²„ì „)
        st.markdown("#### ?• Business Cycle Clock")
        
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
                st.info("Clock 1: ?°ì´???†ìŒ")
        
        # Clock 2: PIT History
        with col2:
            if precomputed is not None and not precomputed.empty:
                pit_data = precomputed[['data_month', 'LEVEL', 'DIRECTION', 'exp2_regime', 'trade_date']].copy()
                pit_data = pit_data.rename(columns={'data_month': 'date', 'exp2_regime': 'ECI', 
                                                     'LEVEL': 'LEVEL', 'DIRECTION': 'DIRECTION'})
                pit_data = pit_data.drop_duplicates(subset=['date'], keep='last').tail(24)
                
                # First values ì¶”ê?
                first_vals_map = provider._first_vals_map.get(country, {})
                pit_data['LEVEL_first'] = pit_data['date'].map(
                    lambda d: first_vals_map.get(d, {}).get('LEVEL', np.nan))
                pit_data['DIRECTION_first'] = pit_data['date'].map(
                    lambda d: first_vals_map.get(d, {}).get('DIRECTION', np.nan))
                
                fig_c2 = plot_business_clock(pit_data, "2. PIT History (Realized)", compare=True)
                st.plotly_chart(fig_c2, use_container_width=True, config={'displayModeBar': False})
            else:
                st.info("Clock 2: ?°ì´???†ìŒ")
        
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
                    current_fresh['LEVEL_first'] = current_fresh['date'].map(
                        lambda d: first_vals_map.get(d, {}).get('LEVEL', np.nan))
                    current_fresh['DIRECTION_first'] = current_fresh['date'].map(
                        lambda d: first_vals_map.get(d, {}).get('DIRECTION', np.nan))
                    
                    fig_c3 = plot_business_clock(current_fresh, "3. Current Snapshot", compare=True)
                    st.plotly_chart(fig_c3, use_container_width=True, config={'displayModeBar': False})
                else:
                    st.info("Clock 3: ?°ì´???†ìŒ")
            else:
                st.info("Clock 3: ?°ì´???†ìŒ")
        
        # 4. ?ì„¸ ?°ì´???Œì´ë¸?(?‘ê¸°)
        with st.expander("?“‹ ?ì„¸ ?°ì´??ë³´ê¸°"):
            # ë°œí‘œ???•ë³´ ?¬í•¨ ì»¬ëŸ¼ ? íƒ
            cols_to_show = ['trade_date', 'realtime_start', 'data_month', 
                           'exp1_regime', 'exp2_regime', 'exp3_regime',
                           'expected_next_data', 'expected_next_release', 'is_missing']
            
            available_cols = [c for c in cols_to_show if c in precomputed.columns]
            display_df = precomputed[available_cols].tail(24).copy()
            
            # ? ì§œ ?¬ë§·??
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
            
            # ì»¬ëŸ¼ëª??œê???
            col_rename = {
                'trade_date': 'ê±°ë˜??,
                'realtime_start': 'ë°œí‘œ??,
                'data_month': '?°ì´?°ì›”',
                'exp1_regime': 'Exp1',
                'exp2_regime': 'Exp2',
                'exp3_regime': 'Exp3',
                'expected_next_data': '?¤ìŒ?ˆìƒ?°ì´??,
                'expected_next_release': '?¤ìŒ?ˆìƒë°œí‘œ??,
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
        ?“Š Market Regime Monitoring Dashboard | 
        Data Source: FRED, Yahoo Finance | 
        Last Updated: {}
    </div>
    """.format(datetime.now().strftime('%Y-%m-%d %H:%M')),
    unsafe_allow_html=True
)
