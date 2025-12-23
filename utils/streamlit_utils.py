# -*- coding: utf-8 -*-
"""
Streamlit 대시보드용 유틸리티
- Crisis Index 계산 (disparity_df_v2 포팅)
- Plotly 차트 생성 함수들
"""
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, List

# =============================================================================
# Crisis Index 계산 (GD_utils.get_data.disparity_df_v2 포팅)
# =============================================================================
def get_data_yahoo_close(ticker: str, start: str = "2000-01-01") -> pd.DataFrame:
    """Yahoo Finance에서 종가 데이터 다운로드"""
    try:
        data = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        if data.empty:
            return pd.DataFrame()
        
        # MultiIndex 처리
        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs('Close', level=0, axis=1) if 'Close' in data.columns.get_level_values(0) else data
        
        # Series인 경우 DataFrame으로 변환
        if isinstance(data, pd.Series):
            data = data.to_frame(name=ticker)
        elif 'Close' in data.columns:
            data = data[['Close']].rename(columns={'Close': ticker})
        else:
            # 첫 번째 컬럼 사용
            data = data.iloc[:, 0].to_frame(name=ticker)
        
        # timezone 제거
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        
        return data
    except Exception as e:
        print(f"[WARN] Failed to download {ticker}: {e}")
        return pd.DataFrame()


def disparity_df_v2(ticker: str, n: int = 120, l: int = 94, u: int = 100) -> pd.DataFrame:
    """
    Crisis Index 계산
    
    Args:
        ticker: Yahoo Finance 티커
        n: EMA 윈도우 (기본 120일)
        l: 하단 임계값 (기본 94)
        u: 상단 임계값 (기본 100)
    
    Returns:
        DataFrame with columns: [ticker, ema, disparity, cut, CX, ...]
    """
    df = get_data_yahoo_close(ticker)
    if df.empty:
        return pd.DataFrame()
    
    col_name = df.columns[0]
    
    # EMA 계산
    df['ema'] = df[col_name].ewm(span=n).mean()
    
    # Disparity = (Close / EMA) * 100
    df['disparity'] = df[col_name] / df['ema'] * u
    
    # Buy/Sell 신호
    df['cut'] = np.nan
    
    # Sell: disparity가 l 아래로 하락
    sell_mask = (df['disparity'].shift(1) >= l) & (df['disparity'] < l)
    df.loc[sell_mask, 'cut'] = 'sell'
    
    # Buy: disparity가 u 위로 상승
    buy_mask = (df['disparity'].shift(1) <= u) & (df['disparity'] > u)
    df.loc[buy_mask, 'cut'] = 'buy'
    
    # Sell 그룹 인덱스
    df['sell'] = np.nan
    df.loc[sell_mask, 'sell'] = range(sell_mask.sum())
    df['sell'] = df['sell'].ffill()
    
    # BarIndex 계산 (각 sell 그룹 내에서의 인덱스)
    def fill_bar_index(group):
        return pd.Series(range(len(group)), index=group.index)
    
    if df['sell'].notna().any():
        df['BarIndex'] = df.groupby('sell').apply(fill_bar_index).reset_index(level=0, drop=True)
    else:
        df['BarIndex'] = range(len(df))
    
    # 볼린저 밴드 하단 계산
    def ema_std(window):
        if len(window) < n:
            return np.nan
        ema_val = df.loc[window.index[-1], 'ema']
        return np.sqrt(((ema_val - window) ** 2).mean())
    
    df['std'] = df[col_name].rolling(n).apply(ema_std, raw=False)
    df['BBdn'] = df['ema'] - (2 * df['std'])
    
    # cut_temp: forward fill
    df['cut_temp'] = df['cut'].ffill()
    
    # 추가 Buy 조건: sell 상태에서 반등 시
    recovery_buy_mask = (
        (df['cut_temp'] == 'sell') &
        (df['disparity'] > l) &
        (df[col_name] > df['BBdn']) &
        (df['BBdn'] > df['BBdn'].shift(1)) &
        (df['BarIndex'] >= 20)
    )
    df.loc[recovery_buy_mask, 'cut'] = 'buy'
    
    # 최종 cut forward fill
    df['cut'] = df['cut'].ffill()
    
    # CX: buy=1, sell=0
    df['CX'] = np.nan
    df.loc[df['cut'] == 'buy', 'CX'] = 1
    df.loc[df['cut'] == 'sell', 'CX'] = 0
    df['CX'] = df['CX'].ffill().fillna(1)  # 기본값 1 (buy)
    
    return df


# =============================================================================
# Plotly 차트 생성 함수들
# =============================================================================
REGIME_COLORS = {
    '팽창': '#2ca02c',
    '회복': '#ffce30',
    '둔화': '#ff7f0e',
    '침체': '#d62728',
    'Cash': '#ffb347',
    'Half': '#9467bd',
    'Skipped': '#f0f0f0'
}

REGIME_LABELS = {
    '팽창': 'Expansion',
    '회복': 'Recovery', 
    '둔화': 'Slowdown',
    '침체': 'Contraction',
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
    누적 수익률 Plotly 차트
    
    Args:
        precomputed: RegimeProvider의 _precomputed_regimes
        prices: 가격 데이터
        ticker: ETF 티커
        country_name: 국가 이름
        start_date: 시작일
        crisis_data: Crisis Index 데이터 (optional)
    
    Returns:
        Plotly Figure
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    if prices.empty or ticker not in prices.columns:
        fig.update_layout(title=f"[{country_name}] 데이터 없음")
        return fig
    
    # 벤치마크 수익률
    bench = prices[ticker].pct_change().fillna(0)
    bench = bench[bench.index >= start_date]
    if bench.empty:
        fig.update_layout(title=f"[{country_name}] 데이터 없음")
        return fig
    
    b_cum = (1 + bench).cumprod()
    b_cum = b_cum / b_cum.iloc[0]
    
    # 벤치마크 라인
    fig.add_trace(
        go.Scatter(
            x=b_cum.index, y=b_cum.values,
            name='Benchmark',
            line=dict(color='silver', dash='dash'),
            hovertemplate='%{x|%Y-%m-%d}<br>Return: %{y:.2%}<extra>Benchmark</extra>'
        ),
        secondary_y=False
    )
    
    # Exp1, Exp2, Exp3 전략 수익률
    colors = {'exp1': '#1f77b4', 'exp2': '#2ca02c', 'exp3': '#d62728'}
    labels = {'exp1': 'Exp1 (First)', 'exp2': 'Exp2 (Fresh)', 'exp3': 'Exp3 (Smart)'}
    
    regime_weights = {
        '팽창': 2.0, '회복': 1.0, '둔화': 0.5, '침체': 0.0,
        'Cash': 0.0, 'Half': 1.0, 'Skipped': 0.0
    }
    
    for exp_col in ['exp1', 'exp2', 'exp3']:
        regime_col = f'{exp_col}_regime'
        if regime_col not in precomputed.columns:
            continue
        
        sub = precomputed[['trade_date', regime_col]].copy()
        sub = sub.set_index('trade_date').reindex(bench.index).ffill()
        sub[regime_col] = sub[regime_col].fillna('Cash')
        
        # Weight 계산
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
    
    # Crisis-Index 음영 및 Disparity 라인
    if crisis_data is not None and not crisis_data.empty and 'CX' in crisis_data.columns:
        crisis_sub = crisis_data[crisis_data.index >= start_date].copy()
        
        if not crisis_sub.empty:
            # CX < 1 구간에 빨간 음영
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
            
            # Disparity 라인 (우측 Y축)
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
    
    # 레이아웃
    fig.update_layout(
        title=f"[{country_name}] 누적 수익률",
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
    국면 스트립 차트 (Timeline 스타일)
    
    Args:
        precomputed: RegimeProvider의 _precomputed_regimes
        crisis_data: Crisis Index 데이터 (optional)
        start_date: 시작일 (optional)
    
    Returns:
        Plotly Figure
    """
    import plotly.express as px
    
    timeline_data = []
    
    # Exp1, Exp2, Exp3 스트립
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
    
    # Crisis-Index 스트립
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
                
                # CX 값에 따른 색상
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
        fig.update_layout(title="국면 스트립 차트 - 데이터 없음", height=200)
        return fig
    
    df_timeline = pd.DataFrame(timeline_data)
    
    # Color 매핑 - REGIME_COLORS 직접 사용 + Crisis-Index 색상 추가
    color_map = {
        '팽창': '#2ca02c',      # 초록
        '회복': '#ffce30',      # 노랑
        '둔화': '#ff7f0e',      # 주황
        '침체': '#d62728',      # 빨강
        'Cash': '#ffb347',     # 연주황
        'Half': '#9467bd',     # 보라
        'Skipped': '#f0f0f0',  # 연회색
        'Buy': '#2ca02c',      # 초록
        'Sell': '#d62728',     # 빨강
        'Weak Sell': '#ff6666', # 연빨강
        'Mixed': '#ffcc66',    # 연주황
        'Unknown': '#cccccc'   # 회색
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
        title="국면 스트립 차트",
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
    Business Cycle Clock (4분면 차트) - Bokeh 스타일 개선 버전
    
    Args:
        df: Level, Momentum, ECI 컬럼이 있는 DataFrame
        title: 차트 제목
        compare: True면 first vs fresh 화살표 표시
    
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    if df is None or df.empty:
        fig.update_layout(title=title, height=380, width=380)
        return fig
    
    # 축 범위 동적 계산 (데이터에 맞게 타이트하게)
    if 'Level' in df.columns and 'Momentum' in df.columns:
        df_valid = df.dropna(subset=['Level', 'Momentum'])
        if not df_valid.empty:
            x_vals = df_valid['Level'].values
            y_vals = df_valid['Momentum'].values
            
            if compare and 'Level_first' in df_valid.columns:
                x_first = df_valid['Level_first'].dropna().values
                y_first = df_valid['Momentum_first'].dropna().values
                if len(x_first) > 0:
                    x_vals = np.concatenate([x_vals, x_first])
                    y_vals = np.concatenate([y_vals, y_first])
            
            # 데이터 범위에 맞게 축 설정 (20% 패딩)
            max_val = max(abs(x_vals).max(), abs(y_vals).max())
            axis_range = max_val * 1.2  # 20% 패딩
            axis_range = max(axis_range, 0.5)  # 최소 0.5 (너무 작으면 안됨)
        else:
            axis_range = 2
    else:
        axis_range = 2
    
    label_pos = axis_range * 0.7
    
    # 4분면 배경 (더 진한 색상)
    # 1사분면 (우상): 팽창 - 녹색
    fig.add_shape(type="rect", x0=0, x1=axis_range, y0=0, y1=axis_range,
                  fillcolor="rgba(44, 160, 44, 0.25)", line=dict(width=0), layer='below')
    # 2사분면 (좌상): 회복 - 노랑
    fig.add_shape(type="rect", x0=-axis_range, x1=0, y0=0, y1=axis_range,
                  fillcolor="rgba(255, 249, 196, 0.4)", line=dict(width=0), layer='below')
    # 3사분면 (좌하): 침체 - 빨강
    fig.add_shape(type="rect", x0=-axis_range, x1=0, y0=-axis_range, y1=0,
                  fillcolor="rgba(255, 230, 230, 0.4)", line=dict(width=0), layer='below')
    # 4사분면 (우하): 둔화 - 주황
    fig.add_shape(type="rect", x0=0, x1=axis_range, y0=-axis_range, y1=0,
                  fillcolor="rgba(255, 243, 224, 0.4)", line=dict(width=0), layer='below')
    
    # 축 라인
    fig.add_hline(y=0, line_color="gray", line_width=1, line_dash="dot")
    fig.add_vline(x=0, line_color="gray", line_width=1, line_dash="dot")
    
    # 데이터 준비
    if 'Level' in df.columns and 'Momentum' in df.columns:
        d = df.dropna(subset=['Level', 'Momentum'])
        x = d['Level'].values
        y = d['Momentum'].values
        
        # Compare 모드: First 포인트 (회색 점)
        if compare and 'Level_first' in d.columns and 'Momentum_first' in d.columns:
            valid_first = d.dropna(subset=['Level_first', 'Momentum_first'])
            if not valid_first.empty:
                fig.add_trace(go.Scatter(
                    x=valid_first['Level_first'], 
                    y=valid_first['Momentum_first'],
                    mode='markers',
                    marker=dict(size=5, color='gray', opacity=0.5),
                    name='First Value',
                    hovertemplate='First<br>P: %{x:.2f}<br>V: %{y:.2f}<extra></extra>'
                ))
                
                # 화살표: First -> Fresh
                for i in range(len(valid_first)):
                    row = valid_first.iloc[i]
                    dist = np.sqrt((row['Level'] - row['Level_first'])**2 + 
                                   (row['Momentum'] - row['Momentum_first'])**2)
                    if dist > 0.1:
                        fig.add_annotation(
                            x=row['Level'], y=row['Momentum'],
                            ax=row['Level_first'], ay=row['Momentum_first'],
                            xref='x', yref='y',
                            axref='x', ayref='y',
                            arrowhead=3, arrowsize=0.8, arrowwidth=1.5,
                            arrowcolor='rgba(220, 20, 60, 0.4)'
                        )
        
        # 경로 라인 (그라데이션 효과 - 청색 계열)
        if len(x) > 1:
            # 시간에 따른 색상 그라데이션
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
        
        # 경로 포인트 (흰색 원 + 네이비 테두리)
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
        
        # 최신 포인트 강조 (빨간색)
        if len(x) > 0:
            fig.add_trace(go.Scatter(
                x=[x[-1]], y=[y[-1]],
                mode='markers',
                marker=dict(size=14, color='red', line=dict(color='white', width=2)),
                name='Latest',
                hovertemplate='<b>Latest</b><br>P: %{x:.2f}<br>V: %{y:.2f}<extra></extra>'
            ))
    
    # 4분면 라벨
    fig.add_annotation(x=label_pos, y=label_pos, text="<b>Expansion</b>", 
                       showarrow=False, font=dict(size=12, color='green'), opacity=0.7)
    fig.add_annotation(x=-label_pos, y=label_pos, text="<b>Recovery</b>", 
                       showarrow=False, font=dict(size=12, color='goldenrod'), opacity=0.7)
    fig.add_annotation(x=-label_pos, y=-label_pos, text="<b>Contraction</b>", 
                       showarrow=False, font=dict(size=12, color='red'), opacity=0.7)
    fig.add_annotation(x=label_pos, y=-label_pos, text="<b>Slowdown</b>", 
                       showarrow=False, font=dict(size=12, color='darkorange'), opacity=0.7)
    
    # 레이아웃 (정사각형 유지)
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14)),
        xaxis=dict(
            range=[-axis_range, axis_range], 
            zeroline=False, 
            showgrid=False,
            title="Level",
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            range=[-axis_range, axis_range], 
            zeroline=False, 
            showgrid=False,
            title="Momentum"
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
    현재 국면 요약 테이블 생성
    
    Args:
        provider: RegimeProvider 인스턴스
        countries: 국가 리스트
    
    Returns:
        요약 DataFrame
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
            'Level': f"{exp2['Level']:.2f}" if exp2 and pd.notna(exp2.get('Level')) else 'N/A',
            'Momentum': f"{exp2['Momentum']:.2f}" if exp2 and pd.notna(exp2.get('Momentum')) else 'N/A'
        })
    
    return pd.DataFrame(rows)


# =============================================================================
# Market Indicators (Fear & Greed, Yield Spread, DXY)
# =============================================================================
def get_fear_greed_data() -> dict:
    """
    CNN Fear & Greed Index 데이터 로드
    
    Returns:
        dict with 'current', 'previous', 'week_ago', 'month_ago', 'year_ago', 'history'
    """
    import requests
    from datetime import datetime, timedelta
    
    result = {
        'current': None,
        'current_text': None,
        'previous': None,
        'week_ago': None,
        'month_ago': None,
        'year_ago': None,
        'history': pd.DataFrame()
    }
    
    try:
        # CNN Fear & Greed JSON API
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # 현재 값
            if 'fear_and_greed' in data:
                fg = data['fear_and_greed']
                result['current'] = fg.get('score', None)
                result['current_text'] = fg.get('rating', None)
                result['previous'] = fg.get('previous_close', None)
            
            # 1주일 전, 1달 전, 1년 전
            if 'fear_and_greed_historical' in data:
                hist = data['fear_and_greed_historical']
                if 'data' in hist:
                    hist_data = hist['data']
                    
                    # DataFrame으로 변환
                    if hist_data:
                        df = pd.DataFrame(hist_data)
                        df['date'] = pd.to_datetime(df['x'], unit='ms')
                        df['FearGreed'] = df['y']
                        df = df.set_index('date')[['FearGreed']]
                        result['history'] = df
                        
                        # 특정 시점 값 추출
                        now = pd.Timestamp.now()
                        week_ago = now - pd.Timedelta(days=7)
                        month_ago = now - pd.Timedelta(days=30)
                        year_ago = now - pd.Timedelta(days=365)
                        
                        if not df.empty:
                            # 가장 가까운 날짜 찾기
                            try:
                                idx_week = df.index.get_indexer([week_ago], method='nearest')[0]
                                result['week_ago'] = df.iloc[idx_week]['FearGreed']
                            except:
                                pass
                            try:
                                idx_month = df.index.get_indexer([month_ago], method='nearest')[0]
                                result['month_ago'] = df.iloc[idx_month]['FearGreed']
                            except:
                                pass
                            try:
                                idx_year = df.index.get_indexer([year_ago], method='nearest')[0]
                                result['year_ago'] = df.iloc[idx_year]['FearGreed']
                            except:
                                pass
        
        return result
        
    except Exception as e:
        print(f"[WARN] Failed to get Fear & Greed: {e}")
        return result


def get_vix_data(start: str = "2020-01-01") -> pd.DataFrame:
    """
    VIX (CBOE Volatility Index) 데이터 로드 - Fear & Greed 실패시 대체용
    
    Args:
        start: 시작일
    
    Returns:
        DataFrame with VIX close price
    """
    try:
        vix = yf.download('^VIX', start=start, progress=False)
        if vix.empty:
            return pd.DataFrame()
        
        # MultiIndex 처리
        if isinstance(vix.columns, pd.MultiIndex):
            vix = vix['Close'] if 'Close' in vix.columns.get_level_values(0) else vix.iloc[:, 0]
        elif 'Close' in vix.columns:
            vix = vix['Close']
        else:
            vix = vix.iloc[:, 0]
        
        # Series to DataFrame
        if isinstance(vix, pd.Series):
            vix = vix.to_frame(name='VIX')
        else:
            vix = vix.iloc[:, 0].to_frame(name='VIX')
        
        # timezone 제거
        if vix.index.tz is not None:
            vix.index = vix.index.tz_localize(None)
        
        return vix
    except Exception as e:
        print(f"[WARN] Failed to download VIX: {e}")
        return pd.DataFrame()


def get_yield_spread(start: str = "2020-01-01") -> pd.DataFrame:
    """
    10년물-3개월물 국채금리 스프레드 (Yahoo Finance only)
    
    Args:
        start: 시작일
    
    Returns:
        DataFrame with DGS10, DGS2, Spread columns
    """
    try:
        tnx = yf.download('^TNX', start=start, progress=False)  # 10년물
        irx = yf.download('^IRX', start=start, progress=False)  # 13주 T-Bill (3개월)
        
        if tnx.empty:
            print("[WARN] TNX data empty")
            return pd.DataFrame()
        
        # MultiIndex 처리 - 10년물
        if isinstance(tnx.columns, pd.MultiIndex):
            tnx_close = tnx['Close'].iloc[:, 0] if 'Close' in tnx.columns.get_level_values(0) else tnx.iloc[:, 0]
        elif 'Close' in tnx.columns:
            tnx_close = tnx['Close']
        else:
            tnx_close = tnx.iloc[:, 0]
        
        # 3개월물 데이터
        if not irx.empty:
            if isinstance(irx.columns, pd.MultiIndex):
                irx_close = irx['Close'].iloc[:, 0] if 'Close' in irx.columns.get_level_values(0) else irx.iloc[:, 0]
            elif 'Close' in irx.columns:
                irx_close = irx['Close']
            else:
                irx_close = irx.iloc[:, 0]
            
            df = pd.DataFrame({
                'DGS10': tnx_close,
                'DGS2': irx_close  # 실제로는 3개월물이지만 컬럼명 유지
            })
        else:
            # IRX 없으면 10년물만 사용하고 추정값
            df = pd.DataFrame({
                'DGS10': tnx_close,
                'DGS2': tnx_close * 0.85
            })
        
        df['Spread'] = df['DGS10'] - df['DGS2']
        df = df.dropna()
        
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        return df
        
    except Exception as e:
        print(f"[WARN] Yahoo Finance yield data failed: {e}")
        return pd.DataFrame()



def get_dxy_data(start: str = "2020-01-01") -> pd.DataFrame:
    """
    Dollar Index (DXY) 데이터 로드
    
    Args:
        start: 시작일
    
    Returns:
        DataFrame with DXY close price
    """
    try:
        dxy = yf.download('DX-Y.NYB', start=start, progress=False)
        if dxy.empty:
            return pd.DataFrame()
        
        # MultiIndex 처리
        if isinstance(dxy.columns, pd.MultiIndex):
            dxy = dxy['Close'] if 'Close' in dxy.columns.get_level_values(0) else dxy.iloc[:, 0]
        elif 'Close' in dxy.columns:
            dxy = dxy['Close']
        else:
            dxy = dxy.iloc[:, 0]
        
        # Series to DataFrame
        if isinstance(dxy, pd.Series):
            dxy = dxy.to_frame(name='DXY')
        else:
            dxy = dxy.iloc[:, 0].to_frame(name='DXY')
        
        # timezone 제거
        if dxy.index.tz is not None:
            dxy.index = dxy.index.tz_localize(None)
        
        return dxy
    except Exception as e:
        print(f"[WARN] Failed to download DXY: {e}")
        return pd.DataFrame()


def create_indicator_gauge(value: float, title: str, min_val: float, max_val: float, 
                           thresholds: Optional[Dict[str, float]] = None,
                           reverse_colors: bool = False) -> go.Figure:
    """
    지표 게이지 차트 생성
    
    Args:
        value: 현재 값
        title: 지표 이름
        min_val: 최소값
        max_val: 최대값
        thresholds: 색상 임계값 딕셔너리 (예: {'low': 15, 'high': 35})
        reverse_colors: True면 높을수록 녹색 (Fear & Greed용)
    
    Returns:
        Plotly Figure (Gauge)
    """
    # NaN/Infinity 검증
    if value is None or pd.isna(value) or np.isinf(value):
        value = (min_val + max_val) / 2  # 기본값: 중간값
    
    # 값 범위 제한
    value = max(min_val, min(max_val, float(value)))
    
    bar_color = "darkblue"
    
    if thresholds is None:
        low = min_val + (max_val - min_val) * 0.33
        high = min_val + (max_val - min_val) * 0.66
    else:
        low = thresholds.get('low', min_val + (max_val - min_val) * 0.33)
        high = thresholds.get('high', min_val + (max_val - min_val) * 0.66)
    
    # 색상 설정 (reverse_colors: 높을수록 녹색)
    if reverse_colors:
        # Fear & Greed 스타일: 낮으면 빨강(Fear), 높으면 녹색(Greed)
        steps = [
            {'range': [min_val, low], 'color': "#f8d7da"},      # 빨강 (Fear)
            {'range': [low, high], 'color': "#fff3cd"},         # 노랑 (Neutral)
            {'range': [high, max_val], 'color': "#d4edda"}      # 녹색 (Greed)
        ]
    else:
        # VIX/Spread 스타일: 낮으면 녹색, 높으면 빨강
        steps = [
            {'range': [min_val, low], 'color': "#d4edda"},
            {'range': [low, high], 'color': "#fff3cd"},
            {'range': [high, max_val], 'color': "#f8d7da"}
        ]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 16}},
        number={'font': {'size': 28}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1},
            'bar': {'color': bar_color},
            'steps': steps,
            'threshold': {
                'line': {'color': "red", 'width': 2},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=30),
        autosize=True
    )
    
    return fig


def create_indicators_chart(vix_df: pd.DataFrame, spread_df: pd.DataFrame, 
                            dxy_df: pd.DataFrame, days: int = 365) -> go.Figure:
    """
    VIX, Yield Spread, DXY 시계열 차트 (서브플롯)
    
    Args:
        vix_df: VIX 데이터
        spread_df: Yield Spread 데이터
        dxy_df: DXY 데이터
        days: 표시할 일수
    
    Returns:
        Plotly Figure
    """
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("VIX (Fear Index)", "10Y-2Y Spread (%)", "Dollar Index (DXY)")
    )
    
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    
    # VIX
    if not vix_df.empty and 'VIX' in vix_df.columns:
        vix_sub = vix_df[vix_df.index >= cutoff]
        fig.add_trace(
            go.Scatter(
                x=vix_sub.index, y=vix_sub['VIX'],
                name='VIX',
                line=dict(color='#d62728', width=2),
                fill='tozeroy',
                fillcolor='rgba(214, 39, 40, 0.1)'
            ),
            row=1, col=1
        )
        # 임계선
        fig.add_hline(y=20, line_dash="dash", line_color="gray", row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="red", row=1, col=1)
    
    # Yield Spread
    if not spread_df.empty and 'Spread' in spread_df.columns:
        spread_sub = spread_df[spread_df.index >= cutoff]
        colors = ['#d62728' if v < 0 else '#2ca02c' for v in spread_sub['Spread']]
        fig.add_trace(
            go.Bar(
                x=spread_sub.index, y=spread_sub['Spread'],
                name='10Y-2Y',
                marker_color=colors
            ),
            row=2, col=1
        )
        # 0선
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, row=2, col=1)
    
    # DXY
    if not dxy_df.empty and 'DXY' in dxy_df.columns:
        dxy_sub = dxy_df[dxy_df.index >= cutoff]
        fig.add_trace(
            go.Scatter(
                x=dxy_sub.index, y=dxy_sub['DXY'],
                name='DXY',
                line=dict(color='#1f77b4', width=2),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.1)'
            ),
            row=3, col=1
        )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        hovermode='x unified',
        margin=dict(l=50, r=30, t=40, b=30)
    )
    
    # Y축 제목
    fig.update_yaxes(title_text="VIX", row=1, col=1)
    fig.update_yaxes(title_text="Spread %", row=2, col=1)
    fig.update_yaxes(title_text="DXY", row=3, col=1)
    
    return fig


# =============================================================================
# Index Returns Table (Phase 2)
# =============================================================================
INDEX_TICKERS = {
    'S&P 500': '^GSPC',
    'NASDAQ': '^IXIC',
    'Dow Jones': '^DJI',
    'KOSPI': '^KS11',
    'Nikkei': '^N225',
    'Shanghai': '000001.SS',
    'DAX': '^GDAXI',
    'FTSE': '^FTSE',
    'CAC 40': '^FCHI',
    'Hang Seng': '^HSI'
}

SECTOR_ETFS = {
    'Technology (XLK)': 'XLK',
    'Financials (XLF)': 'XLF',
    'Healthcare (XLV)': 'XLV',
    'Consumer Disc (XLY)': 'XLY',
    'Consumer Staples (XLP)': 'XLP',
    'Energy (XLE)': 'XLE',
    'Industrials (XLI)': 'XLI',
    'Materials (XLB)': 'XLB',
    'Real Estate (XLRE)': 'XLRE',
    'Utilities (XLU)': 'XLU',
    'Communication (XLC)': 'XLC'
}


def get_index_returns() -> pd.DataFrame:
    """
    주요 지수 수익률 계산 (1D, 1W, 1M, 3M, 6M, 1Y, MTD, YTD)
    
    기준일: 미국 S&P500 기준 가장 최근 종가가 있는 날짜
    1D: 기준일 종가 vs 기준일 전 거래일 종가
    
    Returns:
        DataFrame with index returns across multiple periods
    """
    from datetime import datetime, timedelta
    
    results = []
    all_data = {}  # 모든 지수 데이터 저장
    
    # 2년치 데이터 다운로드
    today = pd.Timestamp.now().normalize()
    start_date = (today - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
    
    # 1. 모든 지수 데이터 수집
    for name, ticker in INDEX_TICKERS.items():
        try:
            data = yf.download(ticker, start=start_date, progress=False)
            if data.empty:
                continue
            
            if isinstance(data.columns, pd.MultiIndex):
                close = data['Close'].iloc[:, 0] if 'Close' in data.columns.get_level_values(0) else data.iloc[:, 0]
            elif 'Close' in data.columns:
                close = data['Close']
            else:
                close = data.iloc[:, 0]
            
            if close.index.tz is not None:
                close.index = close.index.tz_localize(None)
            
            all_data[name] = close
        except Exception as e:
            print(f"[WARN] Failed to get {name}: {e}")
    
    if not all_data:
        return pd.DataFrame(), None
    
    # 2. 기준일 결정: S&P500 기준 가장 최근 종가가 있는 날짜
    if 'S&P 500' in all_data:
        reference_date = all_data['S&P 500'].index[-1]
    else:
        # S&P500이 없으면 가장 많은 지수가 데이터 있는 최근 날짜
        all_dates = pd.concat([pd.Series(d.index) for d in all_data.values()])
        reference_date = all_dates.value_counts().idxmax()
    
    # 3. 기간 정의 (기준일 기준)
    periods = {
        '1D': reference_date - pd.Timedelta(days=1),  # 전 거래일
        '1W': reference_date - pd.Timedelta(weeks=1),
        '1M': reference_date - pd.DateOffset(months=1),
        '3M': reference_date - pd.DateOffset(months=3),
        '6M': reference_date - pd.DateOffset(months=6),
        '1Y': reference_date - pd.DateOffset(years=1),
        'MTD': reference_date.replace(day=1),
        'YTD': reference_date.replace(month=1, day=1)
    }
    
    # 4. 각 지수별 수익률 계산
    for name, close in all_data.items():
        # 기준일에 데이터가 있는지 확인
        if reference_date not in close.index:
            # 기준일에 데이터 없음 (휴장) - 가장 가까운 이전 거래일 사용
            available = close[close.index <= reference_date]
            if available.empty:
                continue
            current_price = available.iloc[-1]
            actual_ref_date = available.index[-1]
        else:
            current_price = close.loc[reference_date]
            actual_ref_date = reference_date
        
        row = {'Index': name, 'Price': current_price}
        
        for period_name, period_start in periods.items():
            try:
                if period_name == '1D':
                    # 1D: 기준일 바로 전 거래일의 종가
                    prev_data = close[close.index < actual_ref_date]
                    if not prev_data.empty:
                        prev_price = prev_data.iloc[-1]
                        # 전 거래일이 기준일 2일 이상 전이면 휴장이었던 것 - NaN 표시
                        days_diff = (actual_ref_date - prev_data.index[-1]).days
                        if days_diff > 4:  # 주말 포함 4일 이상 차이나면 휴장
                            row[period_name] = np.nan
                        else:
                            row[period_name] = (current_price / prev_price - 1) * 100
                    else:
                        row[period_name] = np.nan
                else:
                    # 다른 기간: 해당 시점에 가장 가까운 데이터
                    past_data = close[close.index <= period_start]
                    if not past_data.empty:
                        past_price = past_data.iloc[-1]
                        row[period_name] = (current_price / past_price - 1) * 100
                    else:
                        row[period_name] = np.nan
            except:
                row[period_name] = np.nan
        
        results.append(row)
    
    # =====================================================
    # 추가 지표: 달러인덱스, 장단기 금리차, 시장폭
    # =====================================================
    
    # 1. Dollar Index (DXY)
    try:
        dxy_data = yf.download('DX-Y.NYB', start=start_date, progress=False)
        if not dxy_data.empty:
            if isinstance(dxy_data.columns, pd.MultiIndex):
                dxy_close = dxy_data['Close'].iloc[:, 0]
            else:
                dxy_close = dxy_data['Close'] if 'Close' in dxy_data.columns else dxy_data.iloc[:, 0]
            
            if dxy_close.index.tz is not None:
                dxy_close.index = dxy_close.index.tz_localize(None)
            
            dxy_current = dxy_close.iloc[-1]
            dxy_row = {'Index': '💵 Dollar Index (DXY)', 'Price': dxy_current}
            
            for period_name, period_start in periods.items():
                try:
                    past_data = dxy_close[dxy_close.index <= period_start]
                    if not past_data.empty:
                        dxy_row[period_name] = (dxy_current / past_data.iloc[-1] - 1) * 100
                    else:
                        dxy_row[period_name] = np.nan
                except:
                    dxy_row[period_name] = np.nan
            results.append(dxy_row)
    except Exception as e:
        print(f"[WARN] DXY row failed: {e}")
    
    # 2. 장단기 금리차 (10Y-3M Spread) - 단위: %p (퍼센트포인트)
    try:
        tnx = yf.download('^TNX', start=start_date, progress=False)  # 10Y
        irx = yf.download('^IRX', start=start_date, progress=False)  # 3M
        if not tnx.empty and not irx.empty:
            if isinstance(tnx.columns, pd.MultiIndex):
                tnx_close = tnx['Close'].iloc[:, 0]
                irx_close = irx['Close'].iloc[:, 0]
            else:
                tnx_close = tnx['Close'] if 'Close' in tnx.columns else tnx.iloc[:, 0]
                irx_close = irx['Close'] if 'Close' in irx.columns else irx.iloc[:, 0]
            
            spread = tnx_close - irx_close
            spread_current = spread.iloc[-1]
            spread_row = {'Index': '📉 장단기 금리차 (10Y-3M)', 'Price': spread_current}
            
            for period_name, period_start in periods.items():
                try:
                    past_data = spread[spread.index <= period_start]
                    if not past_data.empty:
                        # 금리차는 변화율이 아니라 차이(delta)로 표시
                        spread_row[period_name] = spread_current - past_data.iloc[-1]
                    else:
                        spread_row[period_name] = np.nan
                except:
                    spread_row[period_name] = np.nan
            results.append(spread_row)
    except Exception as e:
        print(f"[WARN] Spread row failed: {e}")
    
    # 3. 시장폭 (S&P500 52주 범위 내 위치)
    try:
        spy = yf.download('^GSPC', period='1y', progress=False)
        if not spy.empty:
            if isinstance(spy.columns, pd.MultiIndex):
                spy_close = spy['Close'].iloc[:, 0]
                spy_high = spy['High'].iloc[:, 0]
                spy_low = spy['Low'].iloc[:, 0]
            else:
                spy_close = spy['Close']
                spy_high = spy['High']
                spy_low = spy['Low']
            
            high_52w = spy_high.rolling(252, min_periods=1).max()
            low_52w = spy_low.rolling(252, min_periods=1).min()
            position = (spy_close - low_52w) / (high_52w - low_52w) * 100
            
            breadth_current = position.iloc[-1]
            breadth_row = {'Index': '📊 시장폭 (52주 범위 %)', 'Price': breadth_current}
            
            # 시장폭은 변화율이 아닌 현재 위치만 표시, 나머지는 N/A
            for period_name in periods.keys():
                breadth_row[period_name] = np.nan
            results.append(breadth_row)
    except Exception as e:
        print(f"[WARN] Breadth row failed: {e}")
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    return df, reference_date


def get_sector_returns(reference_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    S&P 500 섹터 ETF 수익률 계산 (8개 기간)
    
    Args:
        reference_date: 기준일자 (None이면 자동 결정)
    
    Returns:
        DataFrame with sector returns
    """
    from datetime import datetime, timedelta
    
    results = []
    
    # 2년치 데이터 다운로드
    today = pd.Timestamp.now().normalize()
    start_date = (today - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
    
    # 기준일자가 없으면 XLK (Technology) 기준으로 설정
    if reference_date is None:
        try:
            ref_data = yf.download('XLK', period='5d', progress=False)
            if not ref_data.empty:
                reference_date = ref_data.index[-1]
                if reference_date.tz is not None:
                    reference_date = reference_date.tz_localize(None)
            else:
                reference_date = today
        except:
            reference_date = today
    
    # 기간 정의 (기준일 기준)
    periods = {
        '1D': reference_date - pd.Timedelta(days=1),
        '1W': reference_date - pd.Timedelta(weeks=1),
        '1M': reference_date - pd.DateOffset(months=1),
        '3M': reference_date - pd.DateOffset(months=3),
        '6M': reference_date - pd.DateOffset(months=6),
        '1Y': reference_date - pd.DateOffset(years=1),
        'MTD': reference_date.replace(day=1),
        'YTD': reference_date.replace(month=1, day=1)
    }
    
    for name, ticker in SECTOR_ETFS.items():
        try:
            data = yf.download(ticker, start=start_date, progress=False)
            if data.empty:
                continue
            
            if isinstance(data.columns, pd.MultiIndex):
                close = data['Close'].iloc[:, 0] if 'Close' in data.columns.get_level_values(0) else data.iloc[:, 0]
            elif 'Close' in data.columns:
                close = data['Close']
            else:
                close = data.iloc[:, 0]
            
            if close.index.tz is not None:
                close.index = close.index.tz_localize(None)
            
            current_price = close.iloc[-1]
            
            row = {'Sector': name, 'Price': current_price}
            
            for period_name, period_start in periods.items():
                try:
                    past_data = close[close.index <= period_start]
                    if not past_data.empty:
                        past_price = past_data.iloc[-1]
                        pct_change = (current_price / past_price - 1) * 100
                        row[period_name] = pct_change
                    else:
                        row[period_name] = np.nan
                except:
                    row[period_name] = np.nan
            
            results.append(row)
            
        except Exception as e:
            print(f"[WARN] Failed to get sector {name}: {e}")
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results)


def create_sector_heatmap(sector_df: pd.DataFrame) -> go.Figure:
    """
    섹터 퍼포먼스 히트맵 생성
    
    Args:
        sector_df: get_sector_returns() 결과
    
    Returns:
        Plotly heatmap figure
    """
    if sector_df.empty:
        fig = go.Figure()
        fig.update_layout(title="섹터 데이터 없음")
        return fig
    
    # 히트맵용 데이터 준비 (Price 제외, 수익률 컬럼만 사용)
    sectors = sector_df['Sector'].tolist()
    periods = [col for col in sector_df.columns if col not in ['Sector', 'Price']]
    
    z_data = sector_df[periods].values
    
    # 색상 스케일 (빨강-흰색-녹색)
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=periods,
        y=sectors,
        colorscale='RdYlGn',
        zmid=0,
        text=[[f"{v:.1f}%" if pd.notna(v) else "" for v in row] for row in z_data],
        texttemplate="%{text}",
        textfont={"size": 12},
        hovertemplate='%{y}<br>%{x}: %{z:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="섹터별 퍼포먼스 히트맵",
        xaxis_title="기간",
        yaxis_title="섹터",
        height=400,
        margin=dict(l=100, r=30, t=50, b=30)
    )
    
    return fig


def create_sector_timeseries(days: int = 180) -> go.Figure:
    """
    섹터 ETF 시계열 차트 생성 (YTD 기준 상대 성과)
    
    Args:
        days: 표시할 기간 (일)
    
    Returns:
        Plotly line chart figure
    """
    from datetime import datetime, timedelta
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    fig = go.Figure()
    
    for name, ticker in SECTOR_ETFS.items():
        try:
            data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), progress=False)
            if data.empty:
                continue
            
            if isinstance(data.columns, pd.MultiIndex):
                close = data['Close'].iloc[:, 0]
            else:
                close = data['Close'] if 'Close' in data.columns else data.iloc[:, 0]
            
            if close.index.tz is not None:
                close.index = close.index.tz_localize(None)
            
            # 시작일 대비 수익률 (%)
            returns = (close / close.iloc[0] - 1) * 100
            
            fig.add_trace(go.Scatter(
                x=returns.index,
                y=returns.values,
                mode='lines',
                name=name,
                hovertemplate=f'{name}<br>%{{x|%Y-%m-%d}}<br>수익률: %{{y:.1f}}%<extra></extra>'
            ))
        except Exception as e:
            print(f"[WARN] Sector timeseries failed for {name}: {e}")
    
    fig.update_layout(
        title=f"섹터 ETF 수익률 비교 (최근 {days}일)",
        xaxis_title="날짜",
        yaxis_title="수익률 (%)",
        height=450,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        hovermode='x unified'
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig


def style_returns_dataframe(df: pd.DataFrame) -> 'pd.io.formats.style.Styler':
    """
    수익률 DataFrame에 색상 스타일 적용
    
    Args:
        df: 수익률 DataFrame
    
    Returns:
        Styled DataFrame
    """
    def color_negative_red(val):
        if pd.isna(val) or not isinstance(val, (int, float)):
            return ''
        color = '#d62728' if val < 0 else '#2ca02c'
        return f'color: {color}; font-weight: bold'
    
    # 수치 컬럼만 스타일 적용
    numeric_cols = [col for col in df.columns if col not in ['Index', 'Sector', 'Price']]
    
    styled = df.style.applymap(color_negative_red, subset=numeric_cols)
    
    # 소수점 포맷
    format_dict = {col: '{:.2f}%' for col in numeric_cols}
    if 'Price' in df.columns:
        format_dict['Price'] = '{:,.2f}'
    
    styled = styled.format(format_dict, na_rep='-')
    
    return styled


# =============================================================================
# ETF Recommendations (Phase 2)
# =============================================================================
ETF_RECOMMENDATIONS = {
    '팽창': {
        'assets': ['SPY', 'QQQ', 'IWM', 'VTI'],
        'description': '성장 자산 (주식) 비중 확대',
        'weight': '100%',
        'color': '#2ca02c'
    },
    '회복': {
        'assets': ['SPY', 'IWM', 'VBK', 'XLF'],
        'description': '성장주 + 소형주 비중 확대 (초기 상승 포착)',
        'weight': '100%',
        'color': '#ffce30'
    },
    '둔화': {
        'assets': ['TLT', 'GLD', 'XLP', 'XLU'],
        'description': '채권 + 금 + 방어주로 전환',
        'weight': '50%',
        'color': '#ff7f0e'
    },
    '침체': {
        'assets': ['SHY', 'BIL', 'TIP', 'CASH'],
        'description': '현금 + 단기채 (리스크 회피)',
        'weight': '0%',
        'color': '#d62728'
    },
    'Cash': {
        'assets': ['SHY', 'BIL'],
        'description': '현금성 자산 유지',
        'weight': '0%',
        'color': '#ffb347'
    },
    'Half': {
        'assets': ['SPY', 'TLT'],
        'description': '주식 + 채권 혼합 (균형 전략)',
        'weight': '50%',
        'color': '#9467bd'
    }
}


def get_etf_recommendation(regime: str) -> dict:
    """
    현재 국면에 따른 ETF 추천
    
    Args:
        regime: 현재 국면 (팽창, 회복, 둔화, 침체, Cash, Half)
    
    Returns:
        dict with recommended ETFs, description, weight
    """
    return ETF_RECOMMENDATIONS.get(regime, ETF_RECOMMENDATIONS['Cash'])


def create_etf_recommendation_card(regime: str) -> str:
    """
    ETF 추천 카드 HTML 생성
    
    Args:
        regime: 현재 국면
    
    Returns:
        HTML string for the recommendation card
    """
    rec = get_etf_recommendation(regime)
    etfs = ', '.join(rec['assets'])
    
    html = f"""
    <div style="
        background: linear-gradient(135deg, {rec['color']}22, {rec['color']}11);
        border-left: 4px solid {rec['color']};
        padding: 15px 20px;
        border-radius: 8px;
        margin: 10px 0;
    ">
        <h4 style="margin: 0 0 10px 0; color: {rec['color']};">
            📊 현재 국면: {regime}
        </h4>
        <p style="margin: 5px 0;"><b>추천 ETF:</b> {etfs}</p>
        <p style="margin: 5px 0;"><b>비중:</b> {rec['weight']}</p>
        <p style="margin: 5px 0; font-size: 0.9em; opacity: 0.8;">{rec['description']}</p>
    </div>
    """
    return html


# =============================================================================
# Regime Statistics (Phase 2)
# =============================================================================
def calculate_regime_statistics(precomputed: pd.DataFrame, prices: pd.DataFrame, 
                                  ticker: str, method: str = 'exp2') -> pd.DataFrame:
    """
    국면별 과거 수익률 통계 계산
    
    Args:
        precomputed: provider._precomputed_regimes
        prices: 주가 DataFrame
        ticker: 티커
        method: 'exp1', 'exp2', 'exp3'
    
    Returns:
        DataFrame with regime statistics
    """
    if precomputed is None or precomputed.empty or prices.empty or ticker not in prices.columns:
        return pd.DataFrame()
    
    regime_col = f'{method}_regime'
    if regime_col not in precomputed.columns:
        return pd.DataFrame()
    
    # 데이터 준비
    df = precomputed[['trade_date', regime_col]].copy()
    df = df.set_index('trade_date').sort_index()
    
    # 주가 일별 수익률
    price = prices[ticker].dropna()
    if price.index.tz is not None:
        price.index = price.index.tz_localize(None)
    
    daily_ret = price.pct_change().dropna()
    
    # 국면 데이터와 조인
    combined = pd.DataFrame({'return': daily_ret})
    combined = combined.join(df, how='left')
    combined[regime_col] = combined[regime_col].ffill()
    combined = combined.dropna()
    
    if combined.empty:
        return pd.DataFrame()
    
    # 국면별 통계 계산
    stats = []
    for regime in combined[regime_col].unique():
        regime_data = combined[combined[regime_col] == regime]['return']
        
        if len(regime_data) > 0:
            stats.append({
                '국면': regime,
                '기간(일)': len(regime_data),
                '평균 일수익률': regime_data.mean() * 100,
                '연환산 수익률': regime_data.mean() * 252 * 100,
                '변동성(연환산)': regime_data.std() * np.sqrt(252) * 100,
                '최대 수익': regime_data.max() * 100,
                '최대 손실': regime_data.min() * 100,
                '샤프비율': (regime_data.mean() * 252) / (regime_data.std() * np.sqrt(252)) if regime_data.std() > 0 else 0
            })
    
    return pd.DataFrame(stats)


def create_regime_stats_chart(stats_df: pd.DataFrame) -> go.Figure:
    """
    국면별 통계 바 차트 생성
    
    Args:
        stats_df: calculate_regime_statistics 결과
    
    Returns:
        Plotly bar chart
    """
    if stats_df.empty:
        fig = go.Figure()
        fig.update_layout(title="통계 데이터 없음")
        return fig
    
    regime_colors = {
        '팽창': '#2ca02c', '회복': '#ffce30', '둔화': '#ff7f0e', '침체': '#d62728',
        'Cash': '#ffb347', 'Half': '#9467bd', 'Skipped': '#cccccc'
    }
    
    colors = [regime_colors.get(r, '#888888') for r in stats_df['국면']]
    
    fig = go.Figure(data=[
        go.Bar(
            x=stats_df['국면'],
            y=stats_df['연환산 수익률'],
            marker_color=colors,
            text=[f"{v:.1f}%" for v in stats_df['연환산 수익률']],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="국면별 연환산 수익률",
        xaxis_title="국면",
        yaxis_title="연환산 수익률 (%)",
        height=350,
        showlegend=False
    )
    
    # 0선 추가
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    return fig


# =============================================================================
# Market Breadth (Phase 2)
# =============================================================================
def get_market_breadth() -> dict:
    """
    시장폭 데이터 (S&P 500 기준 52주 신고가/신저가)
    
    Returns:
        dict with 'new_highs', 'new_lows', 'advance_decline'
    """
    result = {
        'new_highs': None,
        'new_lows': None,
        'ratio': None,
        'status': 'N/A'
    }
    
    try:
        # 52주 신고가/신저가 지표 (Yahoo Finance에서 대용)
        # MMTH (% stocks above 200-day MA) as proxy
        spy = yf.download('^GSPC', period='1y', progress=False)
        
        if not spy.empty:
            if isinstance(spy.columns, pd.MultiIndex):
                close = spy['Close'].iloc[:, 0]
                high = spy['High'].iloc[:, 0]
                low = spy['Low'].iloc[:, 0]
            else:
                close = spy['Close']
                high = spy['High']
                low = spy['Low']
            
            # 52주 고가/저가 계산
            high_52w = high.rolling(252).max()
            low_52w = low.rolling(252).min()
            
            current_price = close.iloc[-1]
            latest_52w_high = high_52w.iloc[-1]
            latest_52w_low = low_52w.iloc[-1]
            
            # 현재가가 52주 고가 대비 위치 (%)
            position = (current_price - latest_52w_low) / (latest_52w_high - latest_52w_low) * 100
            
            result['position'] = position
            result['52w_high'] = latest_52w_high
            result['52w_low'] = latest_52w_low
            result['current'] = current_price
            
            if position >= 80:
                result['status'] = '강세 (상위 20%)'
            elif position >= 50:
                result['status'] = '중립 (상위 50%)'
            elif position >= 20:
                result['status'] = '약세 (하위 50%)'
            else:
                result['status'] = '극약세 (하위 20%)'
        
        return result
        
    except Exception as e:
        print(f"[WARN] Market breadth failed: {e}")
        return result


def create_breadth_gauge(breadth_data: dict) -> go.Figure:
    """
    시장폭 게이지 생성
    
    Args:
        breadth_data: get_market_breadth() 결과
    
    Returns:
        Plotly gauge figure
    """
    position = breadth_data.get('position', 50)
    
    # NaN/Infinity 검증
    if position is None or pd.isna(position) or np.isinf(position):
        position = 50  # 기본값: 중간
    position = max(0, min(100, float(position)))
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=position,
        title={'text': "52주 범위 내 위치", 'font': {'size': 14}},
        number={'suffix': '%', 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 20], 'color': "#f8d7da"},
                {'range': [20, 50], 'color': "#fff3cd"},
                {'range': [50, 80], 'color': "#d4edda"},
                {'range': [80, 100], 'color': "#28a745"}
            ]
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=30, r=30, t=50, b=30))
    
    return fig
