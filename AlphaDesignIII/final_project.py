import numpy as np
import pandas as pd
from data import generate_portfolio, find_kpi,setup_proxy
from statsmodels.tsa.stattools import coint


def find_cointegrated_pairs(securities_panel):
    '''
    Function to find out best pairs w.r.t co-integration
    :param securities_panel: panel with closing prices per security
    :return: metrix with pvalue of co-integration
    '''
    n = len(securities_panel.columns)
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = securities_panel.keys
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = securities_panel.iloc[:,i]
            S2 = securities_panel.iloc[:,j]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.01:
                pairs.append((securities_panel.columns[i], securities_panel.columns[j],pvalue))
    return score_matrix, pvalue_matrix, pairs


def zscore(series):
    '''
    Function to find of z-score of a series
    :param series:
    :return: zscore in series
    '''
    return (series - series.mean()) / np.std(series)

def main():
    setup_proxy('mlp')

    # get the list of top 30 stocks and their index
    stock_list = ["AAPL","AXP","BA","CAT","CSCO","CVX",
                  "KO","DD","XOM","GE","GS","HD",
                  "IBM","INTC","JNJ","JPM","MCD","MMM",
                  "MRK","MSFT","NKE","PFE","PG","TRV",
                  "UNH","UTX","V","VZ","WMT","DIS",'^DJI']

    port = generate_portfolio(stock_list,'2010-01-01','2015-01-01')
    pos = port['pos']
    price = port['price']

    # find co-integration of closing prices of the series
    scores, pvalues, pairs = find_cointegrated_pairs(price)

    #based on the data, we will select pair VZ and KO.
    z = zscore(price['VZ'] - price['KO'])

    # Our strategy will be based on z-score value
    for di in z.index:
        # high z-score means it will go back to normal soon
        if z[di] > 1:
            pos['VZ'][di] = 1
            pos['KO'][di] = -1
        #lower z-score means it will go back up soon
        elif z[di] < -1:
            pos['VZ'][di] = -1
            pos['KO'][di] = 1

    # use kpi build in previous courses
    find_kpi(port)


if __name__ == '__main__':
    main()
