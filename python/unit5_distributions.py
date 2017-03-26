from matplotlib.pyplot import *
from matplotlib.finance import quotes_historical_yahoo
import numpy as np
import matplotlib.mlab as mlab
if __name__ == '__main__':

    import os
    os.environ['HTTP_PROXY']="proxy.mlp.com:3128"

    ticker='^GSPC'
    startDate=(1990,1,1)
    endDate=(2015,12,21)
    p = quotes_historical_yahoo(ticker, startDate, endDate,
                                asobject=True, adjusted=True)
    returns = (p.aclose[1:] -p.aclose[:-1])/p.aclose[1:]
    [n,bins,patches] = hist(returns, 200,color='burlywood')
    # Calculate measures of central tendency
    mu = np.mean(returns)
    sigma = np.std(returns)
    # Generate the normal distribution line
    x = mlab.normpdf(bins, mu, sigma)
    plot(bins, x, color='mediumslateblue', lw=4)
    title(ticker+": Distirbution of returns over last 3 years")
    xlabel("Returns")
    ylabel("Frequency")
    show()
