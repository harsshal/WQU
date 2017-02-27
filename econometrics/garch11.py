from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
import statistics as st
import statsmodels.tsa.stattools as sttool

def AR_1(gold,gold2):
    plt.plot(gold2)
    plt.grid()
    plt.title("Gold price evolution from 1978 until 2014")
    plt.xlabel("Period")
    plt.ylabel("USD per troy ounce")
    plt.show()

    lag=gold[:-1] # gold price evolution in the previous period
    gold = gold[1:]

    beta, alpha, r_value, p_value, std_err = stats.linregress(gold, lag)
    print("Beta =",beta, "Alpha = ", alpha) # the estimated parameters

    print("R-squared =", r_value**2) # the coefficient of determination shows that gold price in the previous period explains 86.87 % of today’s evolution

    print("p-value =", p_value)

    forecast_gold=np.exp(0.922226946193*7.143933509+0.428123903567) # the forecast is obtained using the following relation beta*gold price from the  current period + alpha
    print("Forecast gold price =",forecast_gold) # The forecasted gold price for 2015

def arma_select_order(gold):
    sttool.arma_order_select_ic(gold,max_ar=3,max_ma=3,ic=['aic','bic'],trend='nc')

    #when searched in (9,9) space,
    #aic says (0,7) and bic says (1,0)
    #we can apply garch model and see if r-squared improves.

def GARCH11_logL(param, r):
    omega, alpha, beta = param
    r = np.array(r)
    n = len(r)
    s = np.ones(n)*0.01
    s[2] = st.variance(r[0:3])
    for i in range(3, n):
        s[i] = omega + alpha*r[i-1]**2 + beta*(s[i-1])  # GARCH(1,1) model
    logL = -((-np.log(s) - r**2/s).sum())
    return logL

def GARCH11(gold):

    o = optimize.fmin(GARCH11_logL,np.array([.1,.1,.1]), args=(gold,), full_output=1)

    R = np.abs(o[0])
    print()
    print("omega = %.6f\nbeta  = %.6f\nalpha = %.6f\n" % (R[0], R[2], R[1]))


def main():
    gold=[5.264967387, 5.719262046, 6.420808929, 6.129616498,
    5.927725706,
    6.048931247, 5.888268354, 5.759847699,
    5.907675246, 6.100812104, 6.079612778, 5.942326823,
    5.949496062, 5.892362186, 5.840496298, 5.885603906,
    5.951033101, 5.950772752, 5.960670232, 5.802994125,
    5.683885843, 5.629669374, 5.631570141, 5.602266411,
    5.735539506, 5.895283989, 6.014130718, 6.096837563,
    6.403193331, 6.544472839, 6.770743551, 6.879715822,
    7.110304209, 7.359798583, 7.41996794, 7.252216944,
    7.143933509] #logarithm of gold price
    gold2=[193, 305, 615, 459, 375, 424, 361, 317, 368, 446,
    437, 381, 384,
    362, 344, 360, 384, 384, 388, 331, 294, 279,
    279, 271, 310, 363, 409, 444, 604, 695, 872, 972, 1225,
    1572, 1669, 1411, 1266] #gold price

    #AR_1(gold,gold2)
    #arma_select_order(gold)
    GARCH11(gold)

if __name__ == '__main__':
    main()
