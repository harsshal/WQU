import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline

def getData(stocks, start, end):
    from pandas.io.data import DataReader
    from datetime import datetime
    import os
    os.environ['HTTP_PROXY']="proxy.mlp.com:3128"
    
    start = datetime(
                int(start/10000),
                int(start%10000/100),
                int(start%100))
    end = datetime(
                int(end/10000),
                int(end%10000/100),
                int(end%100))
    #stocks = ['ORCL', 'TSLA', 'IBM','YELP', 'MSFT']
    ls_key = 'Adj Close'
    #start = datetime(2014,1,1)
    #end = datetime(2014,3,28)    
    f = DataReader(stocks, 'yahoo',start,end)
    
    cleanData = f.ix[ls_key]
    dataFrame = pd.DataFrame(cleanData)
    return dataFrame
    
df = getData(['^DJT', '^DJI'],20110501,20160501)
df['rDJI'] = df['^DJI']/df['^DJI'].shift()
df['rDJT'] = df['^DJT']/df['^DJT'].shift()
df['pair'] = df['^DJT']/df['^DJI']

df['gain'] = np.where(df['pair'] > df['pair'].shift(),
                      df['pair']-df['pair'].shift(),
                      0)
df['loss'] = np.where(df['pair'] < df['pair'].shift(),
                      df['pair'].shift()-df['pair'],
                      0)

df['avgGain'] = pd.rolling_mean(df[['gain']],14)
df['avgLoss'] = pd.rolling_mean(df[['loss']],14)
df['rs'] = df['avgGain'] / df['avgLoss']
df['rsi'] = 100 - (100/(1+df['rs']))

# applying basic mean reversion
# When >80, more bought => pair gonna come down => Buy Industrial, Sell Transport 
# When <20, more sold => pair gonna come up => Buy Transport, Sell Industrial
df['returns'] = 0
df['returns'] = np.where(df['rsi']>80,df['rDJI']-df['rDJT'],df['returns'])
df['returns'] = np.where(df['rsi']<20,df['rDJT']-df['rDJI'],df['returns'])
df['cReturns'] = df['returns'].cumsum()

df[['pair','rsi','cReturns']].plot(subplots=True)
