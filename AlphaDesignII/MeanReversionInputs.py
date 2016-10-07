__author__ = 'harsshal'
import data

def rsi(price):
    import pandas
    window_length = 2
    # Get the difference in price from previous step
    delta = price.diff()
    # Get rid of the first row, which is NaN since it did not have a previous
    # row to calculate the differences
    delta = delta[1:]

    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    # Calculate the EWMA
    #roll_up1 = pandas.stats.moments.ewma(up, window_length)
    #roll_down1 = pandas.stats.moments.ewma(down.abs(), window_length)
    roll_up1 = up.ewm(com=2,min_periods=0,ignore_na=False,adjust=True).mean()
    roll_down1 = down.abs().ewm(com=2,min_periods=0,ignore_na=False,adjust=True).mean()

    # Calculate the RSI based on EWMA
    RS1 = roll_up1 / roll_down1
    RSI1 = 100.0 - (100.0 / (1.0 + RS1))

    return RSI1

def resample_portfolio(portfolio):
    daily_price = portfolio['price']
    weekly_price = portfolio['price'].resample('W-MON').last()

    daily_sma = daily_price.rolling(window=5,min_periods=1).mean()
    weekly_sma = weekly_price.rolling(window=5,min_periods=1).mean()

    daily_rsi = rsi(daily_price)
    weekly_rsi = rsi(weekly_price)


def main():
    portfolio = data.generate_portfolio(['MSFT'],'20130101','20160101')
    resample_portfolio(portfolio)
    print(portfolio)

if __name__ == '__main__':
    main()
