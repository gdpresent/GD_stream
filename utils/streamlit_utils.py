# -*- coding: utf-8 -*-
"""
Streamlit ?Ä?úÎ≥¥?úÏö© ?†Ìã∏Î¶¨Ìã∞
- Crisis Index Í≥ÑÏÇ∞ (disparity_df_v2 ?¨ÌåÖ)
- Plotly Ï∞®Ìä∏ ?ùÏÑ± ?®Ïàò??
"""
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, List

# =============================================================================
# Crisis Index Í≥ÑÏÇ∞ (GD_utils.get_data.disparity_df_v2 ?¨ÌåÖ)
# =============================================================================
def get_data_yahoo_close(ticker: str, start: str = "2000-01-01") -> pd.DataFrame:
    """Yahoo Finance?êÏÑú Ï¢ÖÍ? ?∞Ïù¥???§Ïö¥Î°úÎìú"""
    try:
        data = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        if data.empty:
            return pd.DataFrame()
        
        # MultiIndex Ï≤òÎ¶¨
        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs('Close', level=0, axis=1) if 'Close' in data.columns.get_level_values(0) else data
        
        # Series??Í≤ΩÏö∞ DataFrame?ºÎ°ú Î≥Ä??
        if isinstance(data, pd.Series):
            data = data.to_frame(name=ticker)
        elif 'Close' in data.columns:
            data = data[['Close']].rename(columns={'Close': ticker})
        else:
            # Ï≤?Î≤àÏß∏ Ïª¨Îüº ?¨Ïö©
            data = data.iloc[:, 0].to_frame(name=ticker)
        
        # timezone ?úÍ±∞
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        
        return data
    except Exception as e:
        print(f"[WARN] Failed to download {ticker}: {e}")
        return pd.DataFrame()


def disparity_df_v2(ticker: str, n: int = 120, l: int = 94, u: int = 100) -> pd.DataFrame:
    """
    Crisis Index Í≥ÑÏÇ∞
    
    Args:
        ticker: Yahoo Finance ?∞Ïª§
        n: EMA ?àÎèÑ??(Í∏∞Î≥∏ 120??
        l: ?òÎã® ?ÑÍ≥ÑÍ∞?(Í∏∞Î≥∏ 94)
        u: ?ÅÎã® ?ÑÍ≥ÑÍ∞?(Í∏∞Î≥∏ 100)
    
    Returns:
        DataFrame with columns: [ticker, ema, disparity, cut, CX, ...]
    """
    df = get_data_yahoo_close(ticker)
    if df.empty:
        return pd.DataFrame()
    
    col_name = df.columns[0]
    
    # EMA Í≥ÑÏÇ∞
    df['ema'] = df[col_name].ewm(span=n).mean()
    
    # Disparity = (Close / EMA) * 100
    df['disparity'] = df[col_name] / df['ema'] * u
    
    # Buy/Sell ?†Ìò∏
    df['cut'] = np.nan
    
    # Sell: disparityÍ∞Ä l ?ÑÎûòÎ°??òÎùΩ
    sell_mask = (df['disparity'].shift(1) >= l) & (df['disparity'] < l)
    df.loc[sell_mask, 'cut'] = 'sell'
    
    # Buy: disparityÍ∞Ä u ?ÑÎ°ú ?ÅÏäπ
    buy_mask = (df['disparity'].shift(1) <= u) & (df['disparity'] > u)
    df.loc[buy_mask, 'cut'] = 'buy'
    
    # Sell Í∑∏Î£π ?∏Îç±??
    df['sell'] = np.nan
    df.loc[sell_mask, 'sell'] = range(sell_mask.sum())
    df['sell'] = df['sell'].ffill()
    
    # BarIndex Í≥ÑÏÇ∞ (Í∞?sell Í∑∏Î£π ?¥Ïóê?úÏùò ?∏Îç±??
    def fill_bar_index(group):
        return pd.Series(range(len(group)), index=group.index)
    
    if df['sell'].notna().any():
        df['BarIndex'] = df.groupby('sell').apply(fill_bar_index).reset_index(level=0, drop=True)
    else:
        df['BarIndex'] = range(len(df))
    
    # Î≥ºÎ¶∞?Ä Î∞¥Îìú ?òÎã® Í≥ÑÏÇ∞
    def ema_std(window):
        if len(window) < n:
            return np.nan
        ema_val = df.loc[window.index[-1], 'ema']
        return np.sqrt(((ema_val - window) ** 2).mean())
    
    df['std'] = df[col_name].rolling(n).apply(ema_std, raw=False)
    df['BBdn'] = df['ema'] - (2 * df['std'])
    
    # cut_temp: forward fill
    df['cut_temp'] = df['cut'].ffill()
    
    # Ï∂îÍ? Buy Ï°∞Í±¥: sell ?ÅÌÉú?êÏÑú Î∞òÎì± ??
    recovery_buy_mask = (
        (df['cut_temp'] == 'sell') &
        (df['disparity'] > l) &
        (df[col_name] > df['BBdn']) &
        (df['BBdn'] > df['BBdn'].shift(1)) &
        (df['BarIndex'] >= 20)
    )
    df.loc[recovery_buy_mask, 'cut'] = 'buy'
    
    # ÏµúÏ¢Ö cut forward fill
    df['cut'] = df['cut'].ffill()
    
    # CX: buy=1, sell=0
    df['CX'] = np.nan
    df.loc[df['cut'] == 'buy', 'CX'] = 1
    df.loc[df['cut'] == 'sell', 'CX'] = 0
    df['CX'] = df['CX'].ffill().fillna(1)  # Í∏∞Î≥∏Í∞?1 (buy)
    
    return df


# =============================================================================
# Plotly Ï∞®Ìä∏ ?ùÏÑ± ?®Ïàò??
# =============================================================================
REGIME_COLORS = {
    '?ΩÏ∞Ω': '#2ca02c',
    '?åÎ≥µ': '#ffce30',
    '?îÌôî': '#ff7f0e',
    'Ïπ®Ï≤¥': '#d62728',
    'Cash': '#ffb347',
    'Half': '#9467bd',
    'Skipped': '#f0f0f0'
}

REGIME_LABELS = {
    '?ΩÏ∞Ω': 'Expansion',
    '?åÎ≥µ': 'Recovery', 
    '?îÌôî': 'Slowdown',
    'Ïπ®Ï≤¥': 'Contraction',
    'Cash': 'Cash',
    'Half': 'Half',
    'Skipped': 'Skipped'
}


def plot_cumulative_returns(
    precomputed: pd.DataFrame,
    prices: pd.DataFrame,
    ticker: str,
    country_name: str,
    start_date: pd.Timestamp,
    crisis_data: Optional[pd.DataFrame] = None
) -> go.Figure:
    """
    ?ÑÏ†Å ?òÏùµÎ•?Plotly Ï∞®Ìä∏
    
    Args:
        precomputed: RegimeProvider??_precomputed_regimes
        prices: Í∞ÄÍ≤??∞Ïù¥??
        ticker: ETF ?∞Ïª§
        country_name: Íµ?? ?¥Î¶Ñ
        start_date: ?úÏûë??
        crisis_data: Crisis Index ?∞Ïù¥??(optional)
    
    Returns:
        Plotly Figure
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    if prices.empty or ticker not in prices.columns:
        fig.update_layout(title=f"[{country_name}] ?∞Ïù¥???ÜÏùå")
        return fig
    
    # Î≤§ÏπòÎßàÌÅ¨ ?òÏùµÎ•?
    bench = prices[ticker].pct_change().fillna(0)
    bench = bench[bench.index >= start_date]
    if bench.empty:
        fig.update_layout(title=f"[{country_name}] ?∞Ïù¥???ÜÏùå")
        return fig
    
    b_cum = (1 + bench).cumprod()
    b_cum = b_cum / b_cum.iloc[0]
    
    # Î≤§ÏπòÎßàÌÅ¨ ?ºÏù∏
    fig.add_trace(
        go.Scatter(
            x=b_cum.index, y=b_cum.values,
            name='Benchmark',
            line=dict(color='silver', dash='dash'),
            hovertemplate='%{x|%Y-%m-%d}<br>Return: %{y:.2%}<extra>Benchmark</extra>'
        ),
        secondary_y=False
    )
    
    # Exp1, Exp2, Exp3 ?ÑÎûµ ?òÏùµÎ•?
    colors = {'exp1': '#1f77b4', 'exp2': '#2ca02c', 'exp3': '#d62728'}
    labels = {'exp1': 'Exp1 (First)', 'exp2': 'Exp2 (Fresh)', 'exp3': 'Exp3 (Smart)'}
    
    regime_weights = {
        '?ΩÏ∞Ω': 2.0, '?åÎ≥µ': 1.0, '?îÌôî': 0.5, 'Ïπ®Ï≤¥': 0.0,
        'Cash': 0.0, 'Half': 1.0, 'Skipped': 0.0
    }
    
    for exp_col in ['exp1', 'exp2', 'exp3']:
        regime_col = f'{exp_col}_regime'
        if regime_col not in precomputed.columns:
            continue
        
        sub = precomputed[['trade_date', regime_col]].copy()
        sub = sub.set_index('trade_date').reindex(bench.index).ffill()
        sub[regime_col] = sub[regime_col].fillna('Cash')
        
        # Weight Í≥ÑÏÇ∞
        sub['weight'] = sub[regime_col].map(lambda x: regime_weights.get(x, 0))
        
        st_ret = (1 + sub['weight'] * bench).cumprod()
        st_ret = st_ret / st_ret.iloc[0]
        
        fig.add_trace(
            go.Scatter(
                x=st_ret.index, y=st_ret.values,
                name=labels[exp_col],
                line=dict(color=colors[exp_col], width=2),
                hovertemplate='%{x|%Y-%m-%d}<br>Return: %{y:.2%}<extra>' + labels[exp_col] + '</extra>'
            ),
            secondary_y=False
        )
    
    # Crisis-Index ?åÏòÅ Î∞?Disparity ?ºÏù∏
    if crisis_data is not None and not crisis_data.empty and 'CX' in crisis_data.columns:
        crisis_sub = crisis_data[crisis_data.index >= start_date].copy()
        
        if not crisis_sub.empty:
            # CX < 1 Íµ¨Í∞Ñ??Îπ®Í∞Ñ ?åÏòÅ
            crisis_sub['cx_grp'] = (crisis_sub['CX'] != crisis_sub['CX'].shift()).cumsum()
            
            for grp_id, grp_data in crisis_sub.groupby('cx_grp'):
                cx_val = grp_data['CX'].iloc[0]
                if pd.notna(cx_val) and cx_val < 1:
                    alpha = (1 - cx_val) * 0.3
                    fig.add_vrect(
                        x0=grp_data.index[0], x1=grp_data.index[-1],
                        fillcolor='rgba(255, 102, 102, {})'.format(alpha),
                        layer='below', line_width=0
                    )
            
            # Disparity ?ºÏù∏ (?∞Ï∏° YÏ∂?
            if 'disparity' in crisis_sub.columns:
                disp = crisis_sub['disparity'].dropna()
                if not disp.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=disp.index, y=disp.values,
                            name='Disparity',
                            line=dict(color='#9467bd', width=1.5),
                            opacity=0.7,
                            hovertemplate='%{x|%Y-%m-%d}<br>Disparity: %{y:.1f}<extra>Disparity</extra>'
                        ),
                        secondary_y=True
                    )
    
    # ?àÏù¥?ÑÏõÉ
    fig.update_layout(
        title=f"[{country_name}] ?ÑÏ†Å ?òÏùµÎ•?,
        xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified',
        height=400
    )
    fig.update_yaxes(title_text="Cumulative Return", secondary_y=False)
    fig.update_yaxes(title_text="Disparity", secondary_y=True)
    
    return fig


def plot_regime_strip(
    precomputed: pd.DataFrame,
    crisis_data: Optional[pd.DataFrame] = None,
    start_date: Optional[pd.Timestamp] = None
) -> go.Figure:
    """
    Íµ?©¥ ?§Ìä∏Î¶?Ï∞®Ìä∏ (Timeline ?§Ì???
    
    Args:
        precomputed: RegimeProvider??_precomputed_regimes
        crisis_data: Crisis Index ?∞Ïù¥??(optional)
        start_date: ?úÏûë??(optional)
    
    Returns:
        Plotly Figure
    """
    import plotly.express as px
    
    timeline_data = []
    
    # Exp1, Exp2, Exp3 ?§Ìä∏Î¶?
    for exp_col in ['exp1_regime', 'exp2_regime', 'exp3_regime']:
        if exp_col not in precomputed.columns:
            continue
        
        exp_name = exp_col.replace('_regime', '').upper()
        sub = precomputed[['trade_date', exp_col]].copy()
        sub = sub.dropna(subset=[exp_col])
        if sub.empty:
            continue
        
        sub['next_date'] = sub['trade_date'].shift(-1).fillna(pd.Timestamp.now())
        
        for _, row in sub.iterrows():
            regime = row[exp_col]
            timeline_data.append({
                'Task': exp_name,
                'Start': row['trade_date'],
                'Finish': row['next_date'],
                'Regime': regime,
                'Color': REGIME_COLORS.get(regime, '#cccccc')
            })
    
    # Crisis-Index ?§Ìä∏Î¶?
    if crisis_data is not None and not crisis_data.empty and 'CX' in crisis_data.columns:
        crisis_sub = crisis_data.copy()
        if start_date is not None:
            crisis_sub = crisis_sub[crisis_sub.index >= start_date]
        
        if not crisis_sub.empty:
            crisis_sub['cx_grp'] = (crisis_sub['CX'] != crisis_sub['CX'].shift()).cumsum()
            
            for grp_id, grp_data in crisis_sub.groupby('cx_grp'):
                cx_val = grp_data['CX'].iloc[0]
                start = grp_data.index[0]
                end = grp_data.index[-1]
                
                # CX Í∞íÏóê ?∞Î•∏ ?âÏÉÅ
                if pd.isna(cx_val):
                    color = '#cccccc'
                    regime = 'Unknown'
                elif cx_val <= 0:
                    color = '#d62728'
                    regime = 'Sell'
                elif cx_val < 0.5:
                    color = '#ff6666'
                    regime = 'Weak Sell'
                elif cx_val < 1.0:
                    color = '#ffcc66'
                    regime = 'Mixed'
                else:
                    color = '#2ca02c'
                    regime = 'Buy'
                
                timeline_data.append({
                    'Task': 'Crisis-Index',
                    'Start': start,
                    'Finish': end + pd.Timedelta(days=1),
                    'Regime': regime,
                    'Color': color
                })
    
    if not timeline_data:
        fig = go.Figure()
        fig.update_layout(title="Íµ?©¥ ?§Ìä∏Î¶?Ï∞®Ìä∏ - ?∞Ïù¥???ÜÏùå", height=200)
        return fig
    
    df_timeline = pd.DataFrame(timeline_data)
    
    # Color Îß§Ìïë - REGIME_COLORS ÏßÅÏ†ë ?¨Ïö© + Crisis-Index ?âÏÉÅ Ï∂îÍ?
    color_map = {
        '?ΩÏ∞Ω': '#2ca02c',      # Ï¥àÎ°ù
        '?åÎ≥µ': '#ffce30',      # ?∏Îûë
        '?îÌôî': '#ff7f0e',      # Ï£ºÌô©
        'Ïπ®Ï≤¥': '#d62728',      # Îπ®Í∞ï
        'Cash': '#ffb347',     # ?∞Ï£º??
        'Half': '#9467bd',     # Î≥¥Îùº
        'Skipped': '#f0f0f0',  # ?∞Ìöå??
        'Buy': '#2ca02c',      # Ï¥àÎ°ù
        'Sell': '#d62728',     # Îπ®Í∞ï
        'Weak Sell': '#ff6666', # ?∞Îπ®Í∞?
        'Mixed': '#ffcc66',    # ?∞Ï£º??
        'Unknown': '#cccccc'   # ?åÏÉâ
    }
    
    fig = px.timeline(
        df_timeline, 
        x_start='Start', 
        x_end='Finish', 
        y='Task',
        color='Regime',
        color_discrete_map=color_map,
        hover_data=['Start', 'Finish', 'Regime']
    )
    
    fig.update_layout(
        title="Íµ?©¥ ?§Ìä∏Î¶?Ï∞®Ìä∏",
        xaxis_title="Date",
        yaxis_title="",
        height=250,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(categoryorder='array', categoryarray=['Crisis-Index', 'EXP3', 'EXP2', 'EXP1'])
    )
    
    return fig


def plot_business_clock(
    df: pd.DataFrame,
    title: str,
    compare: bool = False
) -> go.Figure:
    """
    Business Cycle Clock (4Î∂ÑÎ©¥ Ï∞®Ìä∏) - Bokeh ?§Ì???Í∞úÏÑ† Î≤ÑÏ†Ñ
    
    Args:
        df: LEVEL, DIRECTION, ECI Ïª¨Îüº???àÎäî DataFrame
        title: Ï∞®Ìä∏ ?úÎ™©
        compare: TrueÎ©?first vs fresh ?îÏÇ¥???úÏãú
    
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    if df is None or df.empty:
        fig.update_layout(title=title, height=380, width=380)
        return fig
    
    # Ï∂?Î≤îÏúÑ ?ôÏ†Å Í≥ÑÏÇ∞ (?∞Ïù¥?∞Ïóê ÎßûÍ≤å ?Ä?¥Ìä∏?òÍ≤å)
    if 'LEVEL' in df.columns and 'DIRECTION' in df.columns:
        df_valid = df.dropna(subset=['LEVEL', 'DIRECTION'])
        if not df_valid.empty:
            x_vals = df_valid['LEVEL'].values
            y_vals = df_valid['DIRECTION'].values
            
            if compare and 'LEVEL_first' in df_valid.columns:
                x_first = df_valid['LEVEL_first'].dropna().values
                y_first = df_valid['DIRECTION_first'].dropna().values
                if len(x_first) > 0:
                    x_vals = np.concatenate([x_vals, x_first])
                    y_vals = np.concatenate([y_vals, y_first])
            
            # ?∞Ïù¥??Î≤îÏúÑ??ÎßûÍ≤å Ï∂??§Ï†ï (20% ?®Îî©)
            max_val = max(abs(x_vals).max(), abs(y_vals).max())
            axis_range = max_val * 1.2  # 20% ?®Îî©
            axis_range = max(axis_range, 0.5)  # ÏµúÏÜå 0.5 (?àÎ¨¥ ?ëÏúºÎ©??àÎê®)
        else:
            axis_range = 2
    else:
        axis_range = 2
    
    label_pos = axis_range * 0.7
    
    # 4Î∂ÑÎ©¥ Î∞∞Í≤Ω (??ÏßÑÌïú ?âÏÉÅ)
    # 1?¨Î∂ÑÎ©?(?∞ÏÉÅ): ?ΩÏ∞Ω - ?πÏÉâ
    fig.add_shape(type="rect", x0=0, x1=axis_range, y0=0, y1=axis_range,
                  fillcolor="rgba(44, 160, 44, 0.25)", line=dict(width=0), layer='below')
    # 2?¨Î∂ÑÎ©?(Ï¢åÏÉÅ): ?åÎ≥µ - ?∏Îûë
    fig.add_shape(type="rect", x0=-axis_range, x1=0, y0=0, y1=axis_range,
                  fillcolor="rgba(255, 249, 196, 0.4)", line=dict(width=0), layer='below')
    # 3?¨Î∂ÑÎ©?(Ï¢åÌïò): Ïπ®Ï≤¥ - Îπ®Í∞ï
    fig.add_shape(type="rect", x0=-axis_range, x1=0, y0=-axis_range, y1=0,
                  fillcolor="rgba(255, 230, 230, 0.4)", line=dict(width=0), layer='below')
    # 4?¨Î∂ÑÎ©?(?∞Ìïò): ?îÌôî - Ï£ºÌô©
    fig.add_shape(type="rect", x0=0, x1=axis_range, y0=-axis_range, y1=0,
                  fillcolor="rgba(255, 243, 224, 0.4)", line=dict(width=0), layer='below')
    
    # Ï∂??ºÏù∏
    fig.add_hline(y=0, line_color="gray", line_width=1, line_dash="dot")
    fig.add_vline(x=0, line_color="gray", line_width=1, line_dash="dot")
    
    # ?∞Ïù¥??Ï§ÄÎπ?
    if 'LEVEL' in df.columns and 'DIRECTION' in df.columns:
        d = df.dropna(subset=['LEVEL', 'DIRECTION'])
        x = d['LEVEL'].values
        y = d['DIRECTION'].values
        
        # Compare Î™®Îìú: First ?¨Ïù∏??(?åÏÉâ ??
        if compare and 'LEVEL_first' in d.columns and 'DIRECTION_first' in d.columns:
            valid_first = d.dropna(subset=['LEVEL_first', 'DIRECTION_first'])
            if not valid_first.empty:
                fig.add_trace(go.Scatter(
                    x=valid_first['LEVEL_first'], 
                    y=valid_first['DIRECTION_first'],
                    mode='markers',
                    marker=dict(size=5, color='gray', opacity=0.5),
                    name='First Value',
                    hovertemplate='First<br>P: %{x:.2f}<br>V: %{y:.2f}<extra></extra>'
                ))
                
                # ?îÏÇ¥?? First -> Fresh
                for i in range(len(valid_first)):
                    row = valid_first.iloc[i]
                    dist = np.sqrt((row['LEVEL'] - row['LEVEL_first'])**2 + 
                                   (row['DIRECTION'] - row['DIRECTION_first'])**2)
                    if dist > 0.1:
                        fig.add_annotation(
                            x=row['LEVEL'], y=row['DIRECTION'],
                            ax=row['LEVEL_first'], ay=row['DIRECTION_first'],
                            xref='x', yref='y',
                            axref='x', ayref='y',
                            arrowhead=3, arrowsize=0.8, arrowwidth=1.5,
                            arrowcolor='rgba(220, 20, 60, 0.4)'
                        )
        
        # Í≤ΩÎ°ú ?ºÏù∏ (Í∑∏Îùº?∞Ïù¥???®Í≥º - Ï≤?Éâ Í≥ÑÏó¥)
        if len(x) > 1:
            # ?úÍ∞Ñ???∞Î•∏ ?âÏÉÅ Í∑∏Îùº?∞Ïù¥??
            for i in range(len(x) - 1):
                color_intensity = int(50 + (i / (len(x) - 1)) * 150)
                color = f'rgb({50}, {50 + color_intensity//2}, {color_intensity + 100})'
                fig.add_trace(go.Scatter(
                    x=[x[i], x[i+1]], y=[y[i], y[i+1]],
                    mode='lines',
                    line=dict(color=color, width=3),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Í≤ΩÎ°ú ?¨Ïù∏??(?∞ÏÉâ ??+ ?§Ïù¥Îπ??åÎëêÎ¶?
        dates = d['date'].dt.strftime('%Y-%m').tolist() if 'date' in d.columns else [f'{i}' for i in range(len(d))]
        regimes = d['ECI'].tolist() if 'ECI' in d.columns else ['N/A'] * len(d)
        
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers',
            marker=dict(size=10, color='white', line=dict(color='navy', width=2)),
            name='Path',
            text=[f"{d}<br>{r}" for d, r in zip(dates, regimes)],
            hovertemplate='%{text}<br>P: %{x:.2f}<br>V: %{y:.2f}<extra></extra>'
        ))
        
        # ÏµúÏã† ?¨Ïù∏??Í∞ïÏ°∞ (Îπ®Í∞Ñ??
        if len(x) > 0:
            fig.add_trace(go.Scatter(
                x=[x[-1]], y=[y[-1]],
                mode='markers',
                marker=dict(size=14, color='red', line=dict(color='white', width=2)),
                name='Latest',
                hovertemplate='<b>Latest</b><br>P: %{x:.2f}<br>V: %{y:.2f}<extra></extra>'
            ))
    
    # 4Î∂ÑÎ©¥ ?ºÎ≤®
    fig.add_annotation(x=label_pos, y=label_pos, text="<b>Expansion</b>", 
                       showarrow=False, font=dict(size=12, color='green'), opacity=0.7)
    fig.add_annotation(x=-label_pos, y=label_pos, text="<b>Recovery</b>", 
                       showarrow=False, font=dict(size=12, color='goldenrod'), opacity=0.7)
    fig.add_annotation(x=-label_pos, y=-label_pos, text="<b>Contraction</b>", 
                       showarrow=False, font=dict(size=12, color='red'), opacity=0.7)
    fig.add_annotation(x=label_pos, y=-label_pos, text="<b>Slowdown</b>", 
                       showarrow=False, font=dict(size=12, color='darkorange'), opacity=0.7)
    
    # ?àÏù¥?ÑÏõÉ (?ïÏÇ¨Í∞ÅÌòï ?†Ï?)
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14)),
        xaxis=dict(
            range=[-axis_range, axis_range], 
            zeroline=False, 
            showgrid=False,
            title="LEVEL",
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            range=[-axis_range, axis_range], 
            zeroline=False, 
            showgrid=False,
            title="DIRECTION"
        ),
        height=380,
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=True,
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        ),
        plot_bgcolor='white'
    )
    
    return fig


def create_regime_summary_table(provider, countries: List[str]) -> pd.DataFrame:
    """
    ?ÑÏû¨ Íµ?©¥ ?îÏïΩ ?åÏù¥Î∏??ùÏÑ±
    
    Args:
        provider: RegimeProvider ?∏Ïä§?¥Ïä§
        countries: Íµ?? Î¶¨Ïä§??
    
    Returns:
        ?îÏïΩ DataFrame
    """
    rows = []
    for country in countries:
        exp1 = provider.get_regime(country, pd.Timestamp.now(), method='first')
        exp2 = provider.get_regime(country, pd.Timestamp.now(), method='fresh')
        exp3 = provider.get_regime(country, pd.Timestamp.now(), method='smart')
        
        rows.append({
            'Country': country,
            'Exp1 (First)': exp1['regime'] if exp1 else 'N/A',
            'Exp2 (Fresh)': exp2['regime'] if exp2 else 'N/A',
            'Exp3 (Smart)': exp3['regime'] if exp3 else 'N/A',
            'Data Month': exp2['data_month'].strftime('%Y-%m') if exp2 and exp2.get('data_month') else 'N/A',
            'LEVEL': f"{exp2['LEVEL']:.2f}" if exp2 and pd.notna(exp2.get('LEVEL')) else 'N/A',
            'DIRECTION': f"{exp2['DIRECTION']:.2f}" if exp2 and pd.notna(exp2.get('DIRECTION')) else 'N/A'
        })
    
    return pd.DataFrame(rows)
