__author__ = 'harsshal'

import data
import pandas as pd
import numpy as np
import scipy.stats.mstats as st

def generate_portfolio():
    # replacing visa with At&t as visa started trading only in 2008
    DJI_constituents = ['MT','JPM','MSFT','DIS','BA','MCD',
                    'TRV','MMM','NKE','UNH','CSCO','JNJ',
                    'T','INTC','KO','PG','MRK','IBM',
                    'DD','UTX','PFE','CAT','GE','VZ',
                    'HD','AXP','GS','AAPL','XOM','CVX']
    close = data.get_yahoo_data(DJI_constituents,'20060901','20160901').fillna(0)
    position = pd.DataFrame(data=0,index=close.index,columns=close.columns)
    portfolio = pd.Panel({'price':close,'pos':position})
    return portfolio

def equal_weighted_portfolio(portfolio):
    """

    :type portfolio: panel
    """
    random_list = []
    while len(np.unique(random_list)) != 10:
        random_list = np.random.random_integers(0,29,10)
    for stock in random_list:
        portfolio['pos'].iloc[:,stock] = round(1000/ portfolio['price'].iloc[0,stock])
    return portfolio

def find_kpi(portfolio):
    price = portfolio['price']
    pos = portfolio['pos']

    capital = price * pos
    pnl = (price - price.shift()) * pos
    cumpnl = pnl.cumsum()
    ret = price.pct_change().fillna(0)

    monthly = ret.resample('M').mean().fillna(0)
    print("\tMonthly returns average : ", monthly.mean())

    print("\tPositive Monthly return percentage :",(monthly[monthly>0]/monthly[monthly>0]).fillna(0).mean())

    yearly = ret.resample('A').mean().fillna(0)
    print("\tYearly returns average : ", yearly.mean())

    monthly_high = cumpnl.resample('M').max().fillna(0)
    monthly_low = cumpnl.resample('M').min().fillna(0)
    print("\tMax monthly Drawdown : ",(monthly_high-monthly_low).max())

    alltime_high = cumpnl.max()
    alltime_low = cumpnl.min()
    print("\tMax Drawdown : ",(alltime_high-alltime_low).max())

    water = (cumpnl.max()-cumpnl).fillna(0).sum()
    earth = (cumpnl - cumpnl.min()).fillna(0).sum()
    print("\tLake ratio : ", water/earth)

    total_returns = ret.sum()
    negative_returns = abs(ret[ret<0]).sum()
    print("\tGain to pain ratio : ",total_returns/negative_returns)


def main():
    portfolio = generate_portfolio()
    eql_wgtd_port = equal_weighted_portfolio(portfolio)
    print("Printing KPIs for equal weighted portfolio : ")
    find_kpi(eql_wgtd_port)

if __name__ == '__main__':
    main()