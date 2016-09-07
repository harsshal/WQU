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