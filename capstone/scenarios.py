__author__ = 'harsshal'
'''
module to implement different scenarios which
try to identify chart patterns and generate signals based on those
'''
import sqlite3
import pandas as pd

def dip_trip(close,sma50,sma100,high10,low10):
    '''
    This scenario try to get advantage of small anomaly in the long term trend
    e.g if you have a few down days in an upwards moving trend
    Inputs needed to identify this pattern are :

    :type close: pandas.core.frame.DataFrame Close prices
    :type sma50: pandas.core.frame.DataFrame 50 day moving averages
    :type sma100: pandas.core.frame.DataFrame 100 day moving average
    :type high10: pandas.core.frame.DataFrame 10 day's high
    :type low10: pandas.core.frame.DataFrame  10 day's high

    :return: dip: pandas.core.frame.DataFrame Portfolio signal

    '''

    buy_dip = ((close > sma50) &
               (close > sma100) &
               (close==low10)
               )
    sell_dip = ((close < sma50) &
               (close < sma100) &
               (close == high10)
               )

    dip = (buy_dip.astype(int) + (sell_dip.astype(int) * -1))
    return dip

def bollinger_finger_finder(close,sma50):
    '''
    This scenario try to identify extreme conditions
    which will trigger in mean reversion in short term horizon

    :type close: pandas.core.frame.DataFrame Close prices
    :type sma50: pandas.core.frame.DataFrame 50 day moving average

    :return: bol_finger : pandas.core.frame.DataFrame Portfolio signal

    '''

    std = close.rolling(50).std()
    upper = sma50 + std * 2
    lower = sma50 - std * 2
    upper_finger = (close>upper)
    lower_finger = (close<lower)
    bol_finger = (upper_finger.astype(int) + (lower_finger.astype(int)* -1))
    return bol_finger

def main():
    '''
    main function in this module to generate difference scenarios
    and dump results in database for future use

    :return: Nothing as results are written in database

    '''

    with sqlite3.connect('alpha.db') as conn:
        conn = sqlite3.connect('alpha.db')
        c = conn.cursor()

        query = 'select * from close'
        close = pd.read_sql_query(query,conn,index_col='date')

        sma50 = close.rolling(50).mean().fillna(0)
        sma100 = close.rolling(100).mean().fillna(0)
        high10 = close.rolling(10).max().fillna(0)
        low10 = close.rolling(10).min().fillna(0)

        dip = dip_trip(close,sma50,sma100,high10,low10)
        c.execute("drop table if exists dip")
        dip.to_sql('dip', conn, if_exists='append',index=True)

        bol_finger = bollinger_finger_finder(close,sma50)
        c.execute("drop table if exists bol_finger")
        bol_finger.to_sql('bol_finger', conn, if_exists='append',index=True)


if __name__ == '__main__':
    main()
