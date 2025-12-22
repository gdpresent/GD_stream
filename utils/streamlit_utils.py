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
        df: PLACEMENT, VELOCITY, ECI 컬럼이 있는 DataFrame
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
    if 'PLACEMENT' in df.columns and 'VELOCITY' in df.columns:
        df_valid = df.dropna(subset=['PLACEMENT', 'VELOCITY'])
        if not df_valid.empty:
            x_vals = df_valid['PLACEMENT'].values
            y_vals = df_valid['VELOCITY'].values
            
            if compare and 'PLACEMENT_first' in df_valid.columns:
                x_first = df_valid['PLACEMENT_first'].dropna().values
                y_first = df_valid['VELOCITY_first'].dropna().values
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
    if 'PLACEMENT' in df.columns and 'VELOCITY' in df.columns:
        d = df.dropna(subset=['PLACEMENT', 'VELOCITY'])
        x = d['PLACEMENT'].values
        y = d['VELOCITY'].values
        
        # Compare 모드: First 포인트 (회색 점)
        if compare and 'PLACEMENT_first' in d.columns and 'VELOCITY_first' in d.columns:
            valid_first = d.dropna(subset=['PLACEMENT_first', 'VELOCITY_first'])
            if not valid_first.empty:
                fig.add_trace(go.Scatter(
                    x=valid_first['PLACEMENT_first'], 
                    y=valid_first['VELOCITY_first'],
                    mode='markers',
                    marker=dict(size=5, color='gray', opacity=0.5),
                    name='First Value',
                    hovertemplate='First<br>P: %{x:.2f}<br>V: %{y:.2f}<extra></extra>'
                ))
                
                # 화살표: First -> Fresh
                for i in range(len(valid_first)):
                    row = valid_first.iloc[i]
                    dist = np.sqrt((row['PLACEMENT'] - row['PLACEMENT_first'])**2 + 
                                   (row['VELOCITY'] - row['VELOCITY_first'])**2)
                    if dist > 0.1:
                        fig.add_annotation(
                            x=row['PLACEMENT'], y=row['VELOCITY'],
                            ax=row['PLACEMENT_first'], ay=row['VELOCITY_first'],
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
            title="PLACEMENT",
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            range=[-axis_range, axis_range], 
            zeroline=False, 
            showgrid=False,
            title="VELOCITY"
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
            'Placement': f"{exp2['placement']:.2f}" if exp2 and pd.notna(exp2.get('placement')) else 'N/A',
            'Velocity': f"{exp2['velocity']:.2f}" if exp2 and pd.notna(exp2.get('velocity')) else 'N/A'
        })
    
    return pd.DataFrame(rows)
