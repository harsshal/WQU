__author__ = 'harsshal'

#from matplotlib.pyplot import *
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def firstTry():
    plot([1,2,3])
    show()
    xlabel('hi guys')

def multipleGraphs(series, lables):
    import matplotlib.pyplot as plt
    import itertools

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    count = len(series)
    colors = itertools.cycle(["r", "b", "g"])
    for s in range(1,count):
        myLable = ''
        if s < len(lables):
            myLable = lables[s]
        else:
            myLable = 'series'+str(s)
        ax1.scatter(series[0], series[s], c=next(colors), marker="s", label= myLable)

    plt.legend(loc='upper left');
    plt.show()

def simpleInterest(P,i,n):
    x = [time for time in range(n)]
    simple = [P*i*time for time in range(n)]
    compound = [P*((1+i)*time - 1) for time in range(n) ]
    multipleGraphs([x,simple, compound],['','simple','compound'])


def getPandasYahooData(ticker, start, end):
    """
    Using pandas to get data from yahoo
    :param ticker: ticker we need data for
    :param start: start date from which we will need data
    :param end: till wjat date we meed data
    :return: dataframe with data
    """

    import os
    os.environ['HTTP_PROXY']="proxy.mlp.com:3128"

    from pandas.io.data import DataReader

    import sys

    try:
        df = DataReader(ticker, "yahoo", start, end)
    except OSError as e:
        print("Got an Error : ", e)
        exit()

    return df

def main():

    import pandas as pd
    import numpy as np
    import math

    indices = input("Enter few indices from https://uk.finance.yahoo.com/intlindices?e=us in CSV format:")
    #indices = '^GSPC,^GSPTSE,^MXX,^BVSP,^MERV'
    index = indices.strip().split(',')

    from datetime import date
    today = date.today()
    pl = getPandasYahooData(index,
    date(today.year-10,today.month,today.day),
    date(today.year,today.month,today.day))

    d = {}
    AdjClose = pl['Adj Close']
    for ii, ind in enumerate(AdjClose.index):
        if ii == 0:
            continue

        # remove NaN with previous values
        for ic, column in enumerate(AdjClose.iloc[ii].index):
            if math.isnan(AdjClose.iloc[ii][ic]) :
                AdjClose.iloc[ii][ic] = AdjClose.iloc[ii-1][ic]

        # Find first entry of the month
        curKey = (ind.year,ind.month)
        if not (curKey in d and d[curKey] < ind.day ):
            d[curKey] = ind.day

    # Clean data to keep only monthly entries
    dropList = []
    for ii, ind in enumerate(AdjClose.index):
        if d[(ind.year,ind.month)] != ind.day:
            dropList.append(ind)
    AdjClose = AdjClose.drop(dropList)
    returns = AdjClose.shift(1) / AdjClose


    cr = returns.corr()
    X,Y = np.meshgrid( range(len(index)), range(len(index)))


    fig = figure()
    ax = Axes3D(fig)
    ax.plot_surface(X,Y,np.array([[cr[i][j] for j in cr] for i in cr]),
                    rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    plt.show()


if __name__ == '__main__':
    main()
