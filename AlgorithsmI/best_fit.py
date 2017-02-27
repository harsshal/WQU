def getMatPlotLibYahooData(ticker, start, end):
    """
    Using matplotlib to get data from yahoo.
    :param ticker: ticker we need data for
    :param start: start date from which we will need data
    :param end: till wjat date we meed data
    :return: dataframe with data
    """

    from matplotlib.finance import _quotes_historical_yahoo
    import matplotlib
    start = (2015,12,15)
    end = (2016,1,14)
    price = matplotlib.finance._quotes_historical_yahoo("IBM",start,end)


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
        df = DataReader(ticker.strip(),  "yahoo", start, end)
    except OSError as e:
        print("Got an Error : ", e)
        exit()

    return df


def myplot(x,y,mylabel):
    """
    function to plot the series
    :param x: x values
    :param y: y values
    :param mylabel: y axis label
    :return: Nothing
    """
    import pylab

    pylab.plot(x , y , label=mylabel)
    pylab.legend()
    pylab.xlabel('x values')
    pylab.show()


def multipleGraphs(series):
    import matplotlib.pyplot as plt
    import itertools

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    count = len(series)
    colors = itertools.cycle(["r", "b", "g"])
    for s in range(1,count):
        ax1.scatter(series[0], series[s],  c=next(colors), marker="s", label='series'+str(s))

    plt.legend(loc='upper left');
    plt.show()


def fitFunc(x,p):
    a,b,c = p
    return a*x**2+b*x+c


def bestFit():
    """
    main function
    :return:
    """

    ticker = input("Please enter a ticker: ").strip()

    df = getPandasYahooData(ticker,'2015-12-15','2016-01-15')

    import scipy.interpolate as ip
    xfine = range(len(df))
    y0 = ip.interp1d(xfine , df['Close'].tolist(),kind='quadratic')
    #myplot(xfine,y0(xfine),'Quadratic Fit')


    import scipy.optimize as spo

    # function inside function so as to give access to df and xfine
    def residuals(p):
        return [ x - fitFunc(y,p)  for x,y in zip(df['Close'].tolist(),xfine)]


    p0 = [1,0,130]

    #traditionnal least squares fit
    pwithout,cov,infodict,mesg,ier=spo.leastsq(residuals, p0,full_output=True)
    #myplot(xfine,[ fitFunc(x,pwithout) for x in xfine],'Least Square')
    multipleGraphs([xfine,[df['Close'][x] for x in xfine],y0(xfine),[ fitFunc(x,pwithout) for x in xfine]])

if __name__ == '__main__':
    bestFit()

