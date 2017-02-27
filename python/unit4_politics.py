import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.ticker as ticker
import matplotlib.finance as finance
from datetime import datetime
import sys

def annualBeta():
    '''
    exercise.
    This function was in the handouts and looked very interesting.
    So I implemented it to see the inner workings.
    Purpose of this function is to calculate beta of a stock.
    :return:
    '''
    def ret_f(ticker,startDate, endDate):
        p = finance.quotes_historical_yahoo(ticker, startDate,
                    endDate,asobject=True,adjusted=True)
        return((p.aclose[1:]-p.aclose[0:-1])/p.aclose[:-1])

    startDate=(1990,1,1)
    endDate=(2014,12,31)

    # Pandas Series for Oracle's Data
    y0=pd.Series(ret_f('ORCL',startDate,endDate))

    # Pandas Series for S&P500 Data
    x0=pd.Series(ret_f('^GSPC',startDate,endDate))

    # Historical Date values of S&P500
    dateVal=finance.quotes_historical_yahoo('^GSPC', startDate,
                        endDate,asobject=True,adjusted=True).date[0:-1]
    lag_year=dateVal[0].strftime("%Y")
    y1,x1,beta,index0=[],[],[],[]

    # Calculate Beta for each year
    for i in range(1,len(dateVal)):
        year=dateVal[i].strftime("%Y")
        if(year==lag_year):
            x1.append(x0[i])
            y1.append(y0[i])
        else:
            model=pd.ols(y=pd.Series(y1),x=pd.Series(x1))
            print(lag_year, round(model.beta[0],4))
            beta.append(model.beta[0])
            index0.append(lag_year)
            x1=[]
            y1=[]
            lag_year=year

    # Plot the main graph
    plt.plot(beta,c='firebrick',label='ORCL Beta w.r.t S&P500')
    plt.hlines(y=1,xmin=0,xmax=25, label='Perfect Correlation',lw=2,color='steelblue')
    plt.legend()
    plt.show()


def weekendDiff () :
    '''
    exercise.
    This function was in the handouts and looked very interesting.
    So I implemented it to see the inner workings.
    purpose of this function is to see weekend effect on the prices.
    :return:
    '''
    tickerName = 'CSCO'
    begdate= datetime.date(2014,1,1)
    enddate = datetime.date.today()

    price = finance.fetch_historical_yahoo(tickerName, begdate,enddate)
    r = mlab.csv2rec(price); price.close()
    r.sort()
    r = r[-30:]  # get the last 30 days
    fig, ax= plt.subplots()
    ax.plot(r.date, r.adj_close, 'o-',color='steelblue')
    ax.set_title('Normal Representation:'+tickerName+' Plot with Weekend gaps')
    fig.autofmt_xdate()
    N = len(r)
    ind = np.arange(N)  # the evenly spaced    plot indices
    def formatDate(x, pos=None):
        thisind = np.clip(int(x+0.5), 0, N-1)
        return r.date[thisind].strftime('%Y-%m-%d')
    fig, ax = plt.subplots()
    ax.plot(ind, r.adj_close, 'd-',color='lightsalmon',lw=2)
    ax.set_title('New Representation:'+tickerName+' Evenlyspaced out points')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(formatDate))
    fig.autofmt_xdate()
    plt.show()


def getPandasYahooData(ticker, start, end):
    """
    Using pandas to get data from yahoo
    :param ticker: ticker we need data for
    :param start: start date from which we will need data
    :param end: till wjat date we meed data
    :return: dataframe with data
    """

    from pandas.io.data import DataReader

    try:
        df = DataReader(ticker, "yahoo", start, end)
    except OSError as e:
        print("Got an Error : ", e)
        exit()

    return df


def findPoliticalInfo(file, yearSeries):
    '''
    helper function which returns president and his party for given series of years.
    :param file: file containing presidents information
    :param yearSeries: series containing years
    :return: series of presidents and parties in same order as year
    '''
    presidents = pd.read_csv(file)
    pr = presidents[presidents['PresidencyStartYear']>=1890][['President'
        ,'Party','PresidencyStartYear','PresidencyEndYear']]
    prList = []
    partyList = []
    for year in yearSeries:
        for prIndex,record in pr.iterrows():
            if record['PresidencyStartYear'] <= year and year < record['PresidencyEndYear']:
                prList.append(record['President'])
                partyList.append(record['Party'])
                break
    return prList,partyList


def politicsEffect():
    '''
    to see the effect of political party on stock market index.
    There are lots of factors which affect the price together.
    Thus we cannot make conclusion based ona single factor.
    :return:
    '''
    #Only has data since 1950 and 1985
    # df = getPandasYahooData(['^GSPC','^DJI'],date(1920,1,1),date(2015,12,31))

    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Correct usage : Script indices.csv presidents.csv [proxy]")
        return
    if len(sys.argv) == 3 :
        script, indexFile, presidentsFile = sys.argv
    else:
        script, indexFile, presidentsFile, proxy = sys.argv

    ind = pd.read_csv(indexFile)
    ind['SPCreturns']=ind['SPC']/ind['SPC'].shift(1)
    ind['DJIreturns']=ind['DJI']/ind['DJI'].shift(1)
    ind['President'],ind['Party'] = findPoliticalInfo(presidentsFile,ind['date'])

    indgroupParty = ind[['President','SPCreturns','DJIreturns']].groupby('President').sum()
    indgroupParty.plot(kind='barh')
    plt.show()

    indgroupParty = ind[['Party','SPCreturns','DJIreturns']].groupby('Party').sum()
    indgroupParty.plot(kind='barh')
    plt.show()

    indgroupParty = ind[['Party','SPCreturns','DJIreturns']].groupby('Party')
    print(indgroupParty.describe())


if __name__ == '__main__':
    politicsEffect()
