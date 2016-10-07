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