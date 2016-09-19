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
    close = data.get_yahoo_data(DJI_constituents,'20130901','20160901').fillna(0)
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


def generate_stop_loss(portfolio):
    import random
    price = portfolio['price']
    pos = portfolio['pos']

    max_price = price.rolling(120).max()

    #dataframe to indicate price is too low and thus time to switch
    switch = price[price/max_price < 0.8].fillna(0)

    # find out last portfolio's initial positions
    pos_list = [column for column,value in enumerate(pos.iloc[0,:]) if value>0]

    for date_index in range(pos.shape[0]):

        # find out allowed columns based on switch dataframe
        good_pos = [column for column,value
                    in enumerate(switch.iloc[date_index,:])
                    if switch.iloc[date_index,column] == 0 ]
        random.shuffle(good_pos)

        # correct position list based on good positions
        pos_list = [column for column in pos_list if column in good_pos]

        # remove elements of good position which are already present in current portfolio
        # so that we can keep adding these positions in current portfolio
        good_pos = [column for column in good_pos if column not in pos_list]

        while len(pos_list) != 10 and len(good_pos) != 0 :
            pos_list.append(good_pos.pop())

        # generate new positions
        for stock in pos_list:
            portfolio['pos'].iloc[date_index,stock] = round(1000/ portfolio['price'].iloc[date_index,stock])

    return portfolio

def main():
    portfolio = generate_portfolio()
    eql_wgtd_port = equal_weighted_portfolio(portfolio)
    print("Printing KPIs for equal weighted portfolio : ")
    find_kpi(eql_wgtd_port)
    stop_loss_port = generate_stop_loss(eql_wgtd_port)
    print("Printing KPIs for stop-loss portfolio : ")
    find_kpi(stop_loss_port)

if __name__ == '__main__':
    main()