#%%
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader import data as web
from functools import reduce
from tabulate import tabulate
from matplotlib.ticker import FormatStrFormatter
import warnings

warnings.filterwarnings('ignore')

#%%

def display(data):
	print(tabulate(data, headers = 'keys', tablefmt = 'psql'))
	return

# 1 - Define `tickers` & `company names` for every instrument
stocks      = {'AMC':'AMC Entertainment', 'CLF':'Cleveland-Cliffs',  'GME': 'GameStop', 'DD':'DuPoint de Nemours', 'M' : 'Macys', 'CLNE': 'Clean Energy Fuels Corp.', 'BB':'BlackBerry', 'RNG':'RingCentral', 'WKHS':'Workhorse Group', 'TWNK':'Hostess Brands', 'SENS':'Senseonics', 'SAVA':'Cassava Sciences', 'TX':'Ternium', 'X': 'US Steel Corp'}
bonds       = {'HCA' : 'HCA', 'VRTX' :  'VRTX'}
commodities = {'BTC-USD' : 'Bitcoin', 'DOGE-USD' : 'Dogecoin'}
instruments = {**stocks, **bonds, **commodities}
tickers     = list(instruments.keys())
instruments_data = {}
N = len(tickers)

# 2 - We will look at stock prices over the past years, starting at January 1, 2015
# 01-01-2015 - 16-04-2020
start = datetime.datetime(2017,1,1)
end = datetime.datetime.today()

# 3 - Let's get instruments data based on the tickers.
# First argument is the series we want, second is the source ("yahoo" for Yahoo! Finance), third is the start date, fourth is the end date
for ticker, instrument in instruments.items():
    print("Loading data series for instrument {} with ticker = {}".format(instruments[ticker], ticker))
    instruments_data[ticker] = web.DataReader(ticker, data_source = 'yahoo', start=start, end = end)
    
instruments_data['AMC']
# %%

# 2.3.1 - keep only 'adjusted close' prices
for ticker, instrument in instruments.items():
    instruments_data[ticker] = instruments_data[ticker]["Adj Close"]
    
    
tr_days = [ len(instr) for _, instr in instruments_data.items() ]
tr_days = pd.DataFrame(tr_days, index = tickers, columns = ["Trading Days"])

tr_days.T


# %%
tr_days_stocks_bonds = instruments_data['AMC'].groupby([instruments_data['AMC'].index.year]).agg('count')
tr_days_bitcoin = instruments_data['BTC-USD'].groupby([instruments_data['BTC-USD'].index.year]).agg('count')
tr_days_palladium = instruments_data['DOGE-USD'].groupby([instruments_data['DOGE-USD'].index.year]).agg('count')

tr_days_per_year = pd.DataFrame([tr_days_stocks_bonds, tr_days_bitcoin, tr_days_palladium], index=["Stocks & Bonds", "Bitcoin", "Dogecoin"])

tr_days_per_year
# %%
data = list(instruments_data.values())
data_df = reduce(lambda x, y: pd.merge(x, y, left_index = True, right_index = True, how = 'outer'), data).dropna()
data_df.columns = tickers

data_df
# %%
tr_days_per_year = data_df['AMC'].groupby([data_df['AMC'].index.year]).agg('count')
tr_days_per_year = pd.DataFrame([tr_days_per_year], index = ["All instruments (merged)"])

tr_days_per_year
# %%
fig, ax = plt.subplots(figsize=(12, 8))
data_df.plot(ax = plt.gca(), grid = True)
ax.set_title('Adjusted Close for all instruments')
ax.set_facecolor((0.95, 0.95, 0.99))
ax.grid(c = (0.75, 0.75, 0.99))
# %%

stock_df = data_df.drop(['BTC-USD', 'DOGE-USD'], axis = 1)
simple_returns = stock_df.apply(lambda x: x /x[0] - 1)
simple_returns.plot(grid = True, figsize = (10, 5)).axhline(y = 0, color = "black", lw=2)
# %%

log_returns = data_df.pct_change()
log_returns

# %%
log_returns.plot(grid = True, figsize = (15,10)).axhline(y = 0, color = "black", lw=2)
# %%
APR = log_returns.groupby([log_returns.index.year]).agg('sum')
APR_avg = APR.mean()

APR
# %%
N = np.array(tr_days_per_year.T)
N_total = np.sum(N)
APY = (1 + APR / N)**N-1
APY_avg = (1 + APR_avg/N_total)**N_total - 1

APY
# %%
pd.DataFrame(APY_avg, columns = ['Average APY']).T
# %%
STD = log_returns.groupby([log_returns.index.year]).agg('std') * np.sqrt(N)
STD_avg = STD.mean()
std = log_returns.std()

STD
# %%
pd.DataFrame(STD_avg, columns = ['Average STD']).T
# %%
# configuration
fig, ax = plt.subplots(figsize = (16,12))
ax.set_title(r"Standard Deviation ($\sigma$) of all instruments for all years")
ax.set_facecolor((0.95, 0.95, 0.99))
ax.grid(c = (0.75, 0.75, 0.99))
ax.set_ylabel(r"Standard Deviation $\sigma$")
ax.set_xlabel(r"Years")
STD.plot(ax = plt.gca(),grid = True)

for instr in STD:
  stds = STD[instr]
  years = list(STD.index)
  for year, std in zip(years, stds):
    label = "%.3f"%std
    plt.annotate(label, xy = (year, std), xytext=((-1)*50, 40),textcoords = 'offset points', ha = 'right', va='bottom', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
      arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
# %%
VAR = STD ** 2
VAR_avg = VAR.mean()

VAR
# %%
pd.DataFrame(VAR_avg, columns = ['Average VAR']).T
# %%
# configuration - generate different colors & sizes
c = [y + x for y, x in zip(APY_avg, STD_avg)]
c = list(map(lambda x : x /max(c), c))
s = list(map(lambda x : x * 600, c))


# plot
fig, ax = plt.subplots(figsize = (16,12))
ax.set_title(r"Risk ($\sigma$) vs Return ($APY$) of all  instruments")
ax.set_facecolor((0.95, 0.95, 0.99))
ax.grid(c = (0.75, 0.75, 0.99))
ax.set_xlabel(r"Standard Deviation $\sigma$")
ax.set_ylabel(r"Annualized Percetaneg Yield $APY$ or $R_{effective}$")
ax.scatter(STD_avg, APY_avg, s = s , c = c , cmap = "Blues", alpha = 0.4, edgecolors="grey", linewidth=2)
ax.axhline(y = 0.0,xmin = 0 ,xmax = 5,c = "blue",linewidth = 1.5,zorder = 0,  linestyle = 'dashed')
ax.axvline(x = 0.0,ymin = 0 ,ymax = 40,c = "blue",linewidth = 1.5,zorder = 0,  linestyle = 'dashed')
for idx, instr in enumerate(list(STD.columns)):
  ax.annotate(instr, (STD_avg[idx] + 0.01, APY_avg[idx]))
# %%
instruments = list(log_returns.columns)
instruments
# %%
def visualize_statistic(statistic, title, limit = 0):
  # configuration
  fig, ax = plt.subplots(figsize = (15,8))
  ax.set_facecolor((0.95, 0.95, 0.99))
  ax.grid(c = (0.75, 0.75, 0.99), axis = 'y')
  colors = sns.color_palette('Reds', n_colors = len(statistic))
  # visualize
  barlist = ax.bar(x = np.arange(len(statistic)), height =  statistic)
  for b, c in zip(barlist, colors):
    b.set_color(c)
  ax.axhline(y = limit, xmin = -1 ,xmax = 1,c = "blue",linewidth = 1.5,zorder = 0,  linestyle = 'dashed')

  # configure more
  for i, v in enumerate(statistic):
      ax.text( i - 0.22,v + 0.01 , str(round(v,3)), color = 'blue', fontweight='bold')
  plt.xticks(np.arange(len(statistic)), instruments)
  plt.title(r"{} for every instrument (i) against market (m) S&P500".format(title))
  plt.xlabel(r"Instrument")
  plt.ylabel(r"{} value".format(title))
  plt.show()

def visualize_model(alpha, beta, data, model):
  fig, axs = plt.subplots(5,4, figsize = (14,10),  constrained_layout = True)
  # fig.tight_layout()
  idx = 0
  R_m = data["^GSPC"]
  del data["^GSPC"]
  for a, b, instr in zip(alpha, beta, data):
    i, j = int(idx / 4), idx % 4
    axs[i, j].set_title("Model : {} fitted for '{}'".format(model, instr))
    axs[i, j].set_facecolor((0.95, 0.95, 0.99))
    axs[i, j].grid(c = (0.75, 0.75, 0.99))
    axs[i, j].set_xlabel(r"Market (S&P500) log returns")
    axs[i, j].set_ylabel(r"{} log returns".format(instr))

    R = data[instr]
    y = a + b * R_m
    axs[i, j].scatter(x = R_m, y = R, label = 'Returns'.format(instr))
    axs[i, j].plot(R_m, y ,color = 'red', label = 'CAPM model')
    idx += 1
# %%
# [*] Risk-Free Asset : 13 Week Tbill (^IRX). Get the most recent value
risk_free = web.DataReader('^IRX', data_source = 'yahoo', start = start, end = end)['Adj Close']
risk_free = float(risk_free.tail(1))

print("Risk-Free rate (Daily T-bill) = {}".format(risk_free))
# %%
# [*] Market          : S&P 500 index (^GSPC) | Yahoo Finance for index pricing, '^GSPC' is the underlying for 'SPX' options.


market = web.DataReader('^GSPC', data_source = 'yahoo', start=log_returns.index[0], end=end)['Adj Close']
market = market.rename("^GSPC")
market_log_returns = market.pct_change()
log_return_total = pd.concat([log_returns, market_log_returns], axis = 1).dropna()

# Descriptive statistics
# Return
log_returns_total = pd.concat([log_returns, market_log_returns], axis=1).dropna()
APR_total = log_returns_total.groupby([log_returns_total.index.year]).agg('sum')
APR_avg_total = APR_total.mean()
APR_avg_market = APR_avg_total['^GSPC']
# RISK
STD_total = log_return_total.groupby([log_return_total.index.year]).agg('std') * np.sqrt(N)
STD_avg_total = STD_total.mean()
STD_avg_market = STD_avg_total['^GSPC']
# %%
pd.DataFrame(APR_avg_total, columns = ['Average APR']).T
# %%
# Calculate correlation ρ & R squared R^2 between all instruments (i) & market (m)
corr = log_returns.corrwith(market_log_returns)
r_squared = corr ** 2

pd.DataFrame(r_squared, columns = ["R squared"]).T
# %%
def CAPM():
  # 1 - Calculate average Risk Premium for every instrument  
  # [*]  _
  #     E[R] - R_f
  # [*]   __
  #     E[R_m] - R_f
  APR_premium        = APR_avg - risk_free
  APR_market_premium = APR_avg_market - risk_free

  # 2 - Calculate α, β
  beta  = corr *  STD_avg / STD_avg_market
  alpha = APR_premium - beta * APR_market_premium 
  
  return alpha, beta

alpha, beta = CAPM()

visualize_statistic(alpha.values, "Alpha α")

pd.DataFrame(alpha, columns = ["Average α"]).T
# %%
visualize_statistic(beta.values, "Beta β", limit = 1)

pd.DataFrame(beta, columns = ["Average β"]).T
# %%

visualize_model(alpha/100, beta, data= log_return_total.copy(), model = 'CAPM')
# %%
beta_reg, alpha_reg = np.polyfit(x = log_returns_total['^GSPC'], y = log_returns_total[log_returns.columns], deg=1)

pd.DataFrame(alpha_reg,  index = log_returns.columns ,columns = ["Average α"]).T
# %%
pd.DataFrame(beta_reg,  index = log_returns.columns ,columns = ["Average β"]).T
# %%
visualize_statistic(beta_reg, "Beta β", limit = 1)
# %%
visualize_model(alpha_reg, beta_reg, data = log_returns_total.copy(), model = 'OLS')
# %%
portfolios = {"#1 dummy (risky)" : {"Return E[R]" : 0, "Risk σ" : 0, "Sharpe Ratio SR" : 0},
              "#1 dummy (total)" : {"Return E[R]" : 0, "Risk σ" : 0, "Sharpe Ratio SR" : 0},
              "#2 optimized max sr (risky)" : {"Return E[R]" : 0, "Risk σ" : 0, "Sharpe Ratio SR" : 0},
              "#2 optimized max sr (total)" : {"Return E[R]" : 0, "Risk σ" : 0, "Sharpe Ratio SR" : 0},
              "#2 optimized min σ (risky)" : {"Return E[R]" : 0, "Risk σ" : 0, "Sharpe Ratio SR" : 0},
              "#2 optimized min σ (total)" : {"Return E[R]" : 0, "Risk σ" : 0, "Sharpe Ratio SR" : 0},
              }

# WEIGHTS, RETURN, RISK
cov = APR.cov()
weights = np.array([ 0.45/ 14] * 14 + [ 0.35 / 2] * 2 + [ 0.1 / 2] * 2)
expected_return = np.sum(APR_avg * weights)
expected_risk   = np.sqrt( np.dot(weights.T , np.dot(cov, weights)) )

# RISKY PORTFOLIO
portfolios["#1 dummy (risky)"]["Return E[R]"]     = expected_return
portfolios["#1 dummy (risky)"]["Risk σ"]          = expected_risk
portfolios["#1 dummy (risky)"]["Sharpe Ratio SR"] = (expected_return - risk_free) / expected_risk

# TOTAL PORTFOLIO
total_expected_return = 0.9 * expected_return + 0.1 * risk_free
total_expected_risk   = 0.9 * expected_risk
portfolios["#1 dummy (total)"]["Return E[R]"]     = total_expected_return
portfolios["#1 dummy (total)"]["Risk σ"]          = total_expected_risk
portfolios["#1 dummy (total)"]["Sharpe Ratio SR"] = (total_expected_return - risk_free) / total_expected_risk

portfolios_df = pd.DataFrame(portfolios).T
portfolios_df
# %%
num_portfolios = 10000
generated_portfolios = []

for idx in range(num_portfolios):
	#1 - select random weights for portfolio holdings & rebalance weights sum to 1
	weights = np.array(np.random.random(18))
	weights /= np.sum(weights)

	#2 calculate return, risk, sharpe ratio
	expected_return = np.sum(APR_avg * weights)
	expected_risk = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
	sharpe_ratio = (expected_return - risk_free) / expected_risk

	#3 - store the results
	generated_portfolios.append([expected_return, expected_risk, sharpe_ratio, weights])

# Locate the 2 'special' portfolios 1) maximum sharpe ratio 2) minimum risk
maximum_sr_portfolio = sorted(generated_portfolios, key = lambda x: -x[2])[0]
minimum_risk_portfolio = sorted(generated_portfolios, key = lambda x: x[1])[0]
max_sr = maximum_sr_portfolio[2]

max_sr_weights = pd.DataFrame(maximum_sr_portfolio[3], index = log_returns.columns, columns = ["Optimal Weights  #2 optimized max sr "]).T
min_risk_weights = pd.DataFrame(minimum_risk_portfolio[3], index = log_returns.columns, columns = ["Optimal Weights  #2 optimized min σ "]).T
# %%
max_sr_weights
# %%
min_risk_weights
# %%
# RISKY PORTFOLIOS
portfolios["#2 optimized max sr (risky)"]["Return E[R]"]     = maximum_sr_portfolio[0]
portfolios["#2 optimized max sr (risky)"]["Risk σ"]          = maximum_sr_portfolio[1]
portfolios["#2 optimized max sr (risky)"]["Sharpe Ratio SR"] = (maximum_sr_portfolio[0] - risk_free) / maximum_sr_portfolio[1]
portfolios["#2 optimized min σ (risky)"]["Return E[R]"]      = minimum_risk_portfolio[0]
portfolios["#2 optimized min σ (risky)"]["Risk σ"]           = minimum_risk_portfolio[1]
portfolios["#2 optimized min σ (risky)"]["Sharpe Ratio SR"]  = (minimum_risk_portfolio[0] - risk_free) / minimum_risk_portfolio[1]

# TOTAL PORTFOLIOS
total_expected_return = 0.9 * maximum_sr_portfolio[0] + 0.1 * risk_free
total_expected_risk   = 0.9 * maximum_sr_portfolio[1]
portfolios["#2 optimized max sr (total)"]["Return E[R]"]     = total_expected_return
portfolios["#2 optimized max sr (total)"]["Risk σ"]          = total_expected_risk
portfolios["#2 optimized max sr (total)"]["Sharpe Ratio SR"] = (total_expected_return - risk_free) / total_expected_risk
total_expected_return = 0.9 * minimum_risk_portfolio[0] + 0.1 * risk_free
total_expected_risk   = 0.9 * minimum_risk_portfolio[1]
portfolios["#2 optimized min σ (total)"]["Return E[R]"]      = total_expected_return
portfolios["#2 optimized min σ (total)"]["Risk σ"]           = total_expected_risk
portfolios["#2 optimized min σ (total)"]["Sharpe Ratio SR"]  = (total_expected_return - risk_free) / total_expected_risk

portfolios_df = pd.DataFrame(portfolios).T
portfolios_df
# %%
# plot the 2 optimized portfolios along with the rest 9998
def plot_simulation(CAL = None, INSTRUMENTS = None) :
    fig, ax = plt.subplots(figsize = (18,12))
    ax.set_facecolor((0.95, 0.95, 0.99))
    ax.grid(c = (0.75, 0.75, 0.99))
    # portfolio #1
    ret  = portfolios["#1 dummy (risky)"]["Return E[R]"]
    risk = portfolios["#1 dummy (risky)"]["Risk σ"]
    sr   = (ret - risk_free) / risk
    ax.scatter(risk, ret, marker = (5,1,0),color = 'y',s = 700, label = 'portfolio #1 (dummy)')
    ax.annotate(round(sr, 2), (risk - 0.006,ret + 0.013), fontsize = 20, color = 'black')
    # portfolio #2
    ret, risk, sr = [x[0] for x in generated_portfolios], [x[1] for x in generated_portfolios], [x[2] for x in generated_portfolios]
    ax.scatter(risk, ret, c = sr, cmap = 'viridis', marker = 'o', s = 10, alpha = 0.5)
    ax.scatter(maximum_sr_portfolio[1], maximum_sr_portfolio[0],marker = (5,1,0),color = 'r',s = 700, label = 'portfolio #2 (max sr)')
    ax.annotate(round(maximum_sr_portfolio[2], 2), (maximum_sr_portfolio[1]  - 0.006,maximum_sr_portfolio[0] + 0.013), fontsize = 20, color = 'black')
    ax.scatter(minimum_risk_portfolio[1], minimum_risk_portfolio[0], marker = (5,1,0), color = 'g',s = 700,  label = 'portfolio # (min risk)')
    ax.annotate(round(minimum_risk_portfolio[2], 2), (minimum_risk_portfolio[1]  - 0.006,minimum_risk_portfolio[0] + 0.013), fontsize = 20, color = 'black')
    # CAL?
    if CAL :
        ax.plot(CAL[0], CAL[1], linestyle = '-', color = 'red', label = 'CAL')
    if INSTRUMENTS :
        ax.scatter(STD_avg, APR_avg, s = s , c = c , cmap = "Blues", alpha = 0.4, edgecolors = "grey", linewidth = 2)
        for idx, instr in enumerate(list(STD.columns)):
            sr = round((APR_avg[idx] - risk_free) / STD_avg[idx] , 2)
            ax.annotate(instr, (STD_avg[idx] + 0.01, APR_avg[idx]))
            ax.annotate(sr, (STD_avg[idx] - 0.005 , APR_avg[idx] + 0.015))
    ax.set_title('10000 SIMULATED PORTFOLIOS')
    ax.set_xlabel('Annualized Risk (σ)')
    ax.set_ylabel('Annualized Returns (APR_avg)')
    ax.legend(labelspacing = 1.2)

plot_simulation()
# %%
cal_x  = np.linspace(0.0, 1.2, 50)
cal_y = risk_free + cal_x * max_sr

plot_simulation(CAL = [cal_x, cal_y] , INSTRUMENTS = 'yes')
# %%
A = np.linspace(0, 10, 10)
utility_dummy    = portfolios["#1 dummy (total)"]["Return E[R]"] - 1/2 * A * portfolios["#1 dummy (total)"]["Risk σ"] ** 2
utility_max_sr   = portfolios["#2 optimized max sr (total)"]["Return E[R]"] - 1/2 * A * portfolios["#2 optimized max sr (total)"]["Risk σ"] ** 2
utility_min_risk = portfolios["#2 optimized min σ (total)"]["Return E[R]"] - 1/2 * A * portfolios["#2 optimized min σ (total)"]["Risk σ"] ** 2

fig, ax = plt.subplots(figsize = (18,12))
ax.set_facecolor((0.95, 0.95, 0.99))
ax.grid(c = (0.75, 0.75, 0.99))

# Risk Free
ax.plot(A, [risk_free] * 10, color = 'y', label = 'risk free', linewidth = 4)

# Portfolio #1
ax.scatter(A, utility_dummy, color = 'r',s = 50)
ax.plot(A, utility_dummy, color = 'r', label = 'portfolio #1 (dummy)')

# Portfolio #2 (max sr)
ax.scatter(A, utility_max_sr, color = 'b',s = 50)
ax.plot(A, utility_max_sr, color = 'b', label = 'portfolio #2 (max sr)')

# Portfolio #2 (min risk)
ax.scatter(A, utility_min_risk, color = 'black',s = 50)
ax.plot(A, utility_min_risk, color = 'black', label = 'portfolio #2 (min risk)')

ax.set_title('Utility Function U = E[r] - 1/2 * A * σ ^{2}', fontsize = 20)
ax.set_xlabel('Risk Aversion (A)', fontsize = 16)
ax.set_ylabel('Utility (U)', fontsize = 16)
ax.set_ylim([-2, 0.8])
ax.legend(labelspacing = 1.2)
# %%
portfolio = portfolios["#2 optimized max sr (total)"]
ret       = portfolio['Return E[R]']
risk      = portfolio['Risk σ']
sr        = portfolio['Sharpe Ratio SR']
utility   = ret - 1/2 * 3 * risk ** 2

portfolio = pd.DataFrame([str(round(ret * 100, 2)) + "%", str(round(risk * 100, 2)) + "%", sr, str(round(utility * 100, 2) ) + "%"], index = ['Return E[R]', 'Risk σ', 'Sharpe Ratio SR', 'Utility U'] ,columns = ["Portfolio #2 optimized max sr "]).T
portfolio
# %%
