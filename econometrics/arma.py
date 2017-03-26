import pandas as pd
import statsmodels.tsa.stattools
import statsmodels.graphics.tsaplots
import statsmodels.api as sm
import matplotlib.pyplot as plt

def main():

    # Q1
    df = pd.read_excel('C:\\Users\\hpatil\\Desktop\\fredgraph.xls',skiprows=10)
    print(statsmodels.tsa.stattools.adfuller(df['CSUSHPISA'],maxlag=1))

    #t-stat seems to be higher than 1,5 and 10.
    #So we cannot rejects the hypothesis of gamma = 0.
    #Thus, so possibility of unit roots.

    # Q2
    diff = df['CSUSHPISA']-df['CSUSHPISA'].shift()
    diff = df['CSUSHPISA']
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(311)
    fig = statsmodels.graphics.tsaplots.plot_acf(diff, lags=200, ax=ax1)
    ax2 = fig.add_subplot(312)
    fig = statsmodels.graphics.tsaplots.plot_pacf(diff, lags=200, ax=ax2)
    ax3 = fig.add_subplot(313)
    fig = plt.plot(diff)
    plt.show()

    # from the graphs plotted, it looks like ACF is decreasing very very slowly. And PACF cuts off at 2.
    # Thus, we can conclude this to be a ARIMA(1,0,0) = AR(1) model

    # Q3
    from statsmodels.tsa.arima_model import _arma_predict_out_of_sample
    res = sm.tsa.ARMA( diff.values, (1, 0)).fit()
    params = res.params
    residuals = res.resid
    p = res.k_ar
    q = res.k_ma
    k_exog = res.k_exog
    k_trend = res.k_trend
    steps = 4

    print(_arma_predict_out_of_sample(params, steps, residuals, p, q, k_trend, k_exog, endog=diff.values, exog=None, start=len(diff.values)))
    # This should print out array of 4 next values out of sample.

    # Q4
    # We can add other indices related to home price Index to improve the prediction
    # Those indices will be : First Mortgage Default Index / REIT

if __name__ == '__main__':
    main()
