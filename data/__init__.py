__author__ = 'harsshal'

def get_yahoo_data(tickers,start,end,only_close=1):
    import pandas_datareader.data as web

    panel = web.DataReader(tickers, 'yahoo', start, end)

    if only_close !=1:
        return panel
    else:
        return panel['Adj Close']

def find_trend(series):
    import numpy as np

    ln = len(series)
    x = pd.Series(np.arange(ln))
    regression = pd.ols(y=series, x=x)

    return regression.beta[0]

def generate_portfolio(stock_list,start_date, end_date):
    """
    function which generate portfolio in a standard format
    :param stock_list:['MT','JPM']
    :param start_date:'20130101'
    :param end_date:'20160101'
    :return:port : panel with price and position dataframe
    """
    import pandas as pd

    close = get_yahoo_data(stock_list,start_date,end_date).fillna(0)
    position = pd.DataFrame(data=0,index=close.index,columns=close.columns)
    portfolio = pd.Panel({'price':close,'pos':position})
    return portfolio


def find_kpi(portfolio):
    """
    This function will print KPIs for a portfolio describe in specific format
    :param portfolio: panel with pos and price dataframe
    :return:nothing
    """
    price = portfolio['price']
    pos = portfolio['pos']

    capital = price * pos
    pnl = (price - price.shift()) * pos
    cumpnl = pnl.cumsum()
    ret = (price.pct_change()*pos).fillna(0)

    monthly = ret.resample('M').sum().mean().fillna(0)
    print("\tMonthly returns average : ", monthly.mean())

    individual_month = ret.resample('M').mean().fillna(0)
    print("\tPositive Monthly return percentage :",
          (individual_month[individual_month>0]/
           individual_month[individual_month>0]).fillna(0).mean().mean())

    yearly = ret.resample('A').sum().mean().fillna(0)
    print("\tYearly returns average : ", yearly.mean())

    monthly_high = cumpnl.resample('M').sum().max().fillna(0)
    monthly_low = cumpnl.resample('M').sum().min().fillna(0)
    print("\tMax monthly Drawdown : ",(monthly_high-monthly_low).max())

    alltime_high = cumpnl.sum().max()
    alltime_low = cumpnl.sum().min()
    print("\tMax Drawdown : ",(alltime_high-alltime_low).max())

    water = (cumpnl.sum().max()-cumpnl.sum()).fillna(0).sum()
    earth = (cumpnl.sum() - cumpnl.sum().min()).fillna(0).sum()
    print("\tLake ratio : ", water/earth)

    total_returns = ret.sum().sum()
    negative_returns = abs(ret[ret<0]).sum().sum()
    print("\tGain to pain ratio : ",total_returns/negative_returns)
