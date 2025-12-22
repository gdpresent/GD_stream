import GD_utils as gdu
import pandas as pd
from datetime import datetime
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
from regime_provider import RegimeProvider, COUNTRY_MAP

# %% 데이터 로딩
provider = RegimeProvider(use_cache=True)
countries = provider.loaded_countries

# %% 파라미터 설정
start_date = '2015-01-01'
method = 'smart'  # 'first', 'fresh', 'smart' → exp1_regime, exp2_regime, exp3_regime

# %% 국가별 Regime 시계열 생성 (trade_date 기준)
# _precomputed_regimes에서 직접 가져옴 (trade_date = 발표일 다음 영업일 = 실제 매매일)
regime_col = {'first': 'exp1_regime', 'fresh': 'exp2_regime', 'smart': 'exp3_regime'}[method]

all_regime_dfs = []
all_score_dfs = []

for c in countries:
    precomp = provider._precomputed_regimes.get(c)
    if precomp is not None and not precomp.empty:
        df = precomp[['trade_date', regime_col]].copy()
        df = df.rename(columns={regime_col: c})
        df = df.set_index('trade_date')
        df = df[df.index >= start_date]
        all_regime_dfs.append(df)

# 병합 (trade_date 기준)
regime_df = pd.concat(all_regime_dfs, axis=1).sort_index()
regime_df = regime_df.ffill()  # 발표 사이 날짜는 이전 값 유지

# score 매핑 (팽창=3, 회복=2, 둔화=1, 침체=0, Cash/Skipped=-1)
score_map = {'팽창': 3, '회복': 2, '둔화': 1, '침체': 0, 'Cash': -1, 'Half': -2, 'Skipped': -3}
score_df = regime_df.replace(score_map)

# weight 매핑
weight_map = {'팽창': 2.0, '회복': 1.0, '둔화': 0.5, '침체': 0.0, 'Cash': 0.0, 'Half': 1.0, 'Skipped': 0.0}
weight_df = regime_df.replace(weight_map)


# %% Crisi-Index
# NASDAQ_QX = gdu.get_data.disparity_df_v2('^IXIC')#['disparity'].rename('NASDAQ_QX').apply(lambda x: round(x, 2))  # .loc["2010":]
# SNP500_QX = gdu.get_data.disparity_df_v2('^GSPC')#['disparity'].rename('SNP500_QX').apply(lambda x: round(x, 2))  # .loc["2010":]

USA_QX = gdu.get_data.disparity_df_v2('^GSPC')['CX'].rename('USA_QX')#['disparity'].rename('SNP500_QX').apply(lambda x: round(x, 2))  # .loc["2010":]
Korea_QX = gdu.get_data.disparity_df_v2('^KS11')['cut'].rename('Korea_QX')#['disparity'].rename('NASDAQ_QX').apply(lambda x: round(x, 2))  # .loc["2010":]
China_QX = gdu.get_data.disparity_df_v2('000001.SS')['cut'].rename('China_QX')#['disparity'].rename('NASDAQ_QX').apply(lambda x: round(x, 2))  # .loc["2010":]
Japan_QX = gdu.get_data.disparity_df_v2('^N225')['cut'].rename('Japan_QX')#['disparity'].rename('NASDAQ_QX').apply(lambda x: round(x, 2))  # .loc["2010":]
Germany_QX = gdu.get_data.disparity_df_v2('^GDAXI')['cut'].rename('Germany_QX')#['disparity'].rename('NASDAQ_QX').apply(lambda x: round(x, 2))  # .loc["2010":]
France_QX = gdu.get_data.disparity_df_v2('^FCHI')['cut'].rename('France_QX')#['disparity'].rename('NASDAQ_QX').apply(lambda x: round(x, 2))  # .loc["2010":]
UK_QX = gdu.get_data.disparity_df_v2('^FTSE')['cut'].rename('UK_QX')#['disparity'].rename('NASDAQ_QX').apply(lambda x: round(x, 2))  # .loc["2010":]
India_QX = gdu.get_data.disparity_df_v2('^NSEI')['cut'].rename('India_QX')#['disparity'].rename('NASDAQ_QX').apply(lambda x: round(x, 2))  # .loc["2010":]
Brazil_QX = gdu.get_data.disparity_df_v2('^BVSP')['cut'].rename('Brazil_QX')#['disparity'].rename('NASDAQ_QX').apply(lambda x: round(x, 2))  # .loc["2010":]

QX_df = pd.concat([USA_QX,Korea_QX,China_QX,Japan_QX,Germany_QX,France_QX,UK_QX,India_QX,Brazil_QX], axis=1)
QX_dff = QX_df.loc[regime_df.index]
AA=regime_df.merge(QX_df.loc[regime_df.index], how='left', left_index=True, right_index=True)
AA=AA[sorted(AA.columns)]

# SNP500_QX = disparity_df_v2_stq('^SPX')['disparity'].rename('SNP500_QX').apply(lambda x: round(x, 2))  # .loc["2010":]
# NASDAQ_QX = disparity_df_v2_stq('^NDQ')['disparity'].rename('NASDAQ_QX').apply(lambda x: round(x, 2))  # .loc["2010":]


# %% 티커 매핑
ticker_map = {c: COUNTRY_MAP[c]['ticker'] for c in countries}
Univ = ['USA','Korea','China','Japan','Germany','France','UK','India','Brazil']
ETF_Univ = [ticker_map[x] for x in Univ]
# %% 결과 확인
max_weight = 1.0 # 개별 ETF 최대 비중

score_df = score_df[score_df>0]
w_df = score_df[Univ].div(score_df[Univ].sum(1), axis=0).dropna(how='all', axis=0)

# max_weight 제한 적용
w_df = w_df.clip(upper=max_weight)

# 남는 비중은 CASH로
w_df['CASH'] = 1 - w_df.sum(axis=1)
w_df['CASH'] = w_df['CASH'].clip(lower=0)  # 음수 방지

# 컬럼명 티커로 변경
w_df = w_df.rename(columns=lambda x: ticker_map.get(x, x))

# %% BM Weight 생성
# MSCI ACWI 기준 국가별 시가총액 비중 (2024년 기준, 대략적)
COUNTRY_MKTCAP_WEIGHT = {
    'USA': 0.62,      # 미국 62%
    'Japan': 0.055,   # 일본 5.5%
    'UK': 0.035,      # 영국 3.5%
    'France': 0.028,  # 프랑스 2.8%
    'Germany': 0.022, # 독일 2.2%
    'China': 0.025,   # 중국 2.5%
    'India': 0.020,   # 인도 2.0%
    'Korea': 0.012,   # 한국 1.2%
    'Brazil': 0.008,  # 브라질 0.8%
}

# Univ 기준 정규화
mktcap_weights = pd.Series({c: COUNTRY_MKTCAP_WEIGHT.get(c, 0.01) for c in Univ})
mktcap_weights = mktcap_weights / mktcap_weights.sum()

# BM DataFrame 생성
bm_equal = pd.DataFrame(1/len(Univ), index=w_df.index, columns=ETF_Univ)
bm_equal['CASH'] = 0

bm_mktcap = pd.DataFrame(index=w_df.index, columns=ETF_Univ)
for c in Univ:
    bm_mktcap[ticker_map[c]] = mktcap_weights[c]
bm_mktcap['CASH'] = 0

print("\n=== BM Weights ===")
print(f"Equal: {round(1/len(Univ), 3)} each")
print(f"MktCap: {mktcap_weights.round(3).to_dict()}")

gdu.data = provider._load_price_data()
gdu.data['CASH'] = 1
p_df=gdu.calc_return(w_df, cost=0).rename('Country Rotation')
eq_df=gdu.calc_return(bm_equal, cost=0).rename('EQ')
mkt_df=gdu.calc_return(bm_mktcap, cost=0).rename('MKT')
gdu.report(pd.concat([p_df, eq_df,mkt_df,gdu.data[ETF_Univ]], axis=1).dropna(subset=['Country Rotation'], axis=0))
# gdu.data.stack().merge(w_df.stack(), left_index=True, right_index=True, how='right')
AA=pd.concat([w_df.stack().rename('w'), gdu.data.stack()], axis=1).dropna(subset=['w']).sort_index()

print("=== Regime DataFrame ===")
print(regime_df.tail(10))

print("\n=== Score DataFrame ===")
print(score_df.tail(10))

print("\n=== Weight DataFrame ===")
print(weight_df.tail(10))

print("\n=== Ticker Map ===")
print(ticker_map)
