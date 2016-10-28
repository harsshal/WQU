__author__ = 'harsshal'

import data

def main():
    """
    generate portfolio
    calculate trend lines
        21 and 45 day ema and calculate 0.5ATR for each ( 2 cases)
        determine sizes by % vol model, market money model, multiple tier model ( 3 cases)
        determinze size changes by upright/inverted/reflecting pyramid ( 3 cases)
        assume slippage / brokerage fee for trading
        60% insample to determine above cases, 20% out of sample for KPIs
    :return:
    """
    port = data.generate_portfolio(['MSFT','AAPL','WMT'],'20140101','20160101')
    print(port)

    port['ema_21'] = port['price'].ewm(21).mean()
    port['ema_45'] = port['price'].ewm(45).mean()



if __name__ == '__main__':
    main()