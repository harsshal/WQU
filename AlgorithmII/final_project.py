# Get current constituents of SPX
#
# Get data for 10 years for those constituents
#
# Get constituents of DJIA of past 10 years
#
# Calculate daily returns of all of the above
#
# Design function port(M,X,N,P)
#
# Rank trend lines on moving average M of year for every year,
#
# in 2 groups SPX and that year's DJIA
#
# We will be adding X units of cash for N years / investing X*N amount each year
#
# After N years, we will worst performing P% stocks
#
# and replace with highest potential trending lines stocks
#
# Pass this function to optimizer and get optimal values for 4 parameters

import pandas as pd
close = pd.DataFrame()

def find_trend(series):
    import numpy as np

    ln = len(series)
    x = pd.Series(np.arange(ln))
    regression = pd.ols(y=series, x=x)

    return regression.beta[0]

def portfolio( array):
    [M, X, N, P] = array

    M=int(M)
    X=int(X)
    N=int(N)
    P=P/100
    port_df = pd.DataFrame(index=range(2006, 2016), columns=close.columns).fillna(0)

    return_df = pd.DataFrame(index=range(2006,2016),columns=close.columns).fillna(0)
    for year in return_df.index:
        for ticker in return_df.columns:
            return_df.loc[year,ticker] = \
                close[close.index.year == year][ticker].sort_index()[-1:][0] / \
                close[close.index.year == year][ticker].sort_index()[:1][0]

    ma = close.rolling(M).mean()

    trend_df = pd.DataFrame(index=range(2006,2016),columns=ma.columns).fillna(0)
    for year in trend_df.index:
        for ticker in trend_df.columns:
            data_series = ma[ma.index.year==year][ticker].dropna().values
            if data_series.size == 0:
                trend_df.loc[year, ticker] = 0
            else:
                trend_df.loc[year,ticker] = find_trend(data_series)

    if N > trend_df.shape[0]:
        return 0
    for year in range(trend_df.index[0],trend_df.index[0]+N):
        if year != trend_df.index[0]:
            port_df.loc[year+1] = port_df.loc[year]
        invest = trend_df.loc[year].abs().sort_values(ascending=False)[:X].index
        for ticker in invest:
            if(trend_df.loc[year,ticker]>0):
                port_df.loc[year+1,ticker] += 1
            else:
                port_df.loc[year+1, ticker] -= 1

    for year in range(trend_df.index[0]+N,2015):
        if year != trend_df.index[0]:
            port_df.loc[year + 1] = port_df.loc[year]

        worst = (return_df*port_df).cumsum().loc[year].sort_values()[:int(P*X*N)].index
        port_df.loc[year+1,worst] = 0

        invest = trend_df.loc[year].abs().sort_values(ascending=False)[:int(P*X*N)].index
        for ticker in invest:
            if (trend_df.loc[year, ticker] > 0):
                port_df.loc[year + 1, ticker] += 1
            else:
                port_df.loc[year + 1, ticker] -= 1

    #print(M,X,N,P,(port_df*return_df).cumsum()[-1:].sum(axis=1).values[0])
    return (port_df*return_df).cumsum()[-1:].sum(axis=1).values[0]

def get_yahoo_data(tickers,start,end,only_close=1):
    import pandas_datareader.data as web

    panel = web.DataReader(tickers, 'yahoo', start, end)

    if only_close !=1:
        return panel
    else:
        return panel['Adj Close']

def main():
    gspc = pd.read_csv('index_constituents', header=None, index_col=0, nrows=1)

    global close
    close = get_yahoo_data(gspc.loc['GSPC_current'].tolist(), '20060101', '20160101')

    print(portfolio( [20, 2, 2, 50]))

    # Could not determine how to make use of functions like minimize or techniques like gradient descent
    # as I dont know what does the curve look like. So going to assume that function is steady and
    # will keep changing the parameters one by one

    #from scipy.optimize import minimize
    #print(minimize(portfolio,[10,2,2,50],method='SLSQP',bounds=((10,200),(1,100),(1,10),(1,100))))

    max=0
    for period in range(10,200,30):
        if max < portfolio([period,2,2,50]):
            max_period = period

    max = 0
    for initial_add in range(10, 200, 30):
        for initial_period in range(1,10):
            if max < portfolio([max_period, initial_add, initial_period, 50]):
                max_initial_add = initial_add
                max_initial_period = initial_period

    max = 0
    for rebalance in range(10,100,10):
        if max < portfolio([max_period, max_initial_add, max_initial_period, rebalance]):
            max_rebalance = rebalance

    print("best values of M,X,N and P are %d,%d,%d and %f"%
          (max_period,max_initial_add,max_initial_period,max_rebalance))


if __name__ == '__main__':
    main()