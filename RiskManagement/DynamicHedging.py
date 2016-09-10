__author__ = 'harsshal'

# Get 2 years of data from yahoo for DJIA
# Calculate sigma / volatility from the data
# Lets assume we start trading on Sep 1st, 2014 with contract expiring in Dec.
# Calculate delta from sigma with t = 0.25
# Every day, t will reduce so that in 90 days it will go to 0
# Lets assume that we rollover to March contract on Dec 1st.
# Based on delta, we will be hedging put options with 200*delta contracts long.
# We will calculate total PNL of each quarter based on the closing price of each quarter

"""
Created on Sun Aug 21 13:18:10 2016

@author: lipi
"""
# Import the required libraries and modules
#-------------------------------------------
from matplotlib.finance import _quotes_historical_yahoo
import pandas as pd
import datetime
import mibian
import quandl
import numpy as np
from math import e


#Initialize the variables
#---------------------------------------------
opt_code = '^DJI'
fut_code = 'CHRIS/CME_YM1'

date1 = datetime.date.today() #today
end= date1 #today
start = date1.replace(year=date1.year - 5) #5 years back

#Function Definations
#----------------------------------------------
# Get Future Prices
def get_hist_futures(future_code,start_date,end_date):
    DJIA_Futures = quandl.get(future_code,
                              authtoken = "cvQViZ3mh8gkuANqgTc_",
                              start_date = start_date,
                              end_date = end_date)
    return DJIA_Futures


#Get Option prices
def get_hist_dow_jones_index(option_code,start_date,end_date):
    DJIA_Index = _quotes_historical_yahoo(option_code, start_date,
                                           end_date,asobject=True, adjusted=True)

    return DJIA_Index

#Find a month woth maximum working days
def find_month_with_max_workdays(work_dates,work_years,work_months):
    df = pd.DataFrame(
    {
        "date": work_dates,
        "month": work_months,
        "year": work_years
        }
    )
    #consider the rows between today and 2 years back{
    start_2yr_bck = date1.replace(year=date1.year - 2) #2 years back
    mask = (df['date'] >= start_2yr_bck) & (df['date'] <= end)
    df = df.loc[mask]#}
    #I start trading on 1st day of the trading month
    y_m_d = df.groupby(["year", "month"])["date"].first().reset_index(name='date')
    #Size of the month
    gb = df.groupby(["year", "month"]).size().reset_index(name='count')
    gb["firstday"] = y_m_d ["date"]
    #Find first month with max working days
    maxx = gb.ix[gb['count'].idxmax()]
    print('\nMax working month in last 5 years is:',maxx['year'],'/',maxx['month'])
    return maxx['year'], maxx['month'], maxx["firstday"]


def calculate_hist_volatility(trade_date,index):
    df = pd.DataFrame(
    {
        "date": index.date[1:],
        "perc": (index.close[1:] / index.close[:-1] -1)#percentage change
        }
    )
    #consider the hstorical volatility before the trading date{
    mask = (df['date'] < trade_date)
    df = df.loc[mask]#}
    return np.std(df["perc"])#workout volatility

def pick_price(date,index):
    #Pick price according to Daye key
    for indx, item in enumerate(index):
        if item.date == date:
            strk_prc = item.close
            break
    return strk_prc

def third_friday_of_month(s,dindex):
    exp_day = 0
    return_date = s
    #check if trading day itself is not a third friday
    if s.weekday() == 4 and 14 < s.day < 22:
        exp_day = 1
    else:
        #Option expires on 3rd friday
        for date in dindex.date:
            if date > s:
                if date.weekday() == 4 and 14 < date.day < 22:
                    delta = date - s
                    exp_day = delta.days
                    return_date = date
                    break
    return exp_day,return_date

def find_days_to_expiry(trading_start_date,index):
    expry_day,ret_date = third_friday_of_month(trading_start_date,index)
    return expry_day,ret_date

def find_futures_expiry(oexp_date,dindex):
    exp_months = [3,6,9,12]#March,June,September,December
    for row in dindex:
        if row.date > oexp_date and row.month in exp_months:
            if row.date.weekday() == 4 and 14 < row.date.day < 22:
                mon_date = oexp_date + datetime.timedelta(days=3)
                delta = row.date - mon_date
                print(oexp_date,mon_date)
                exp_day = delta.days
                break
    return exp_day,mon_date


if __name__ == "__main__":
    # Dow Jones Index Download
    dow_jones_index = get_hist_dow_jones_index(opt_code,start,end)
    #Find Month with Max working Days and Buying Option Day
    year,month,start_date = find_month_with_max_workdays(dow_jones_index.date,dow_jones_index.year,dow_jones_index.month)
    #Get volatility for Delta calculation
    volatility = (calculate_hist_volatility(start_date,dow_jones_index))*np.sqrt(252)*100
    #AssumingdDays to expire is 3rd Friday of every month
    days_to_expiry,opt_exp_date = find_days_to_expiry(start_date,dow_jones_index)
    #Spot Ptice of Option expiry day
    spot_price = pick_price(opt_exp_date,dow_jones_index)
    #At-the-Money(Assuming Spot price is same as my option contract strike price )
    strike_price = spot_price

    interest_rate = 1
    number_of_contract = 1
    contract_size = 100

    print("Volatility:",volatility,'%')
    print("Strike Price:",strike_price)
    print("Spot Price:",spot_price)
    print("Days to Option Expiry:",days_to_expiry,"days")
    print("Interest Rate:",interest_rate,'%')
    #Get Put Delta
    c = mibian.BS([spot_price, strike_price, interest_rate,days_to_expiry], volatility=volatility)
    delta_of_put = (c.putDelta*-1)#Short Put delta is positive
    print("Delta of Put option is:",delta_of_put)
    total_put_delta = number_of_contract*contract_size*delta_of_put
    print("Total Delta of options is:",total_put_delta)

    #Determine the kind of delta neutral hedge needed
    print("Since I am in short Put position, I need a Long Delta hedge,I should buy Futures")
    days_to_fut_expiry,next_trade_day = find_futures_expiry(opt_exp_date,dow_jones_index)
    delta_of_futures = e**(interest_rate*days_to_fut_expiry/252)
    print("Delta of Futures contract is:",delta_of_futures)
    #{Delta Neutral Portfolio = n1D1 + n2D2 = 0
    #           Where D1 = Delta value of the original options.
    #                 D2 = Delta value of hedging options.
    #                 n1 = Amount of original options.
    #                 n2 = Amount of hedging options}
    no_of_fut = total_put_delta/delta_of_futures
    print("Number of Futures contract required to hedge the delta of the out option would be"\
                                                        ,no_of_fut,'\n')



    #Get futures
    futures = get_hist_futures(fut_code,next_trade_day,end)

    num = len(futures)
    num_ind = len(dow_jones_index)
    diff = (num_ind - num)
    # Consider Spot Prices Close and Open for Delta calculation
    futures["Spot_Open"] = dow_jones_index.open[diff:]
    futures["Spot_Close"] = dow_jones_index.close[diff:]

    num_fut_list = []
    verdict = []
    number_to_hedge = []
    verdict.append('')
    num_fut_list.append(no_of_fut)
    current_delta = 0
    number_to_hedge.append(0)

    #Calculating delta of futures as movements in futures per day
   # w.r.t movement of underlying stock per day
    for i, row in enumerate(futures.values):

        date = futures.index[i]
        openF = row[0]
        openS = row[8]
        closeF = row[5]
        closeS = row[9]
        prev_no = no_of_fut

        if (closeS-openS) == 0 or (closeF-openF) == 0:
            no_of_fut = no_of_fut
            num_fut_list.append(no_of_fut)
            if no_of_fut < prev_no:
               verdict.append('Sell')
            else:
               verdict.append('Buy')
            number_to_hedge.append(0)
            continue
        else:
            current_delta = (closeF-openF)/(closeS-openS)
        if i == 0:
            prev_delta = current_delta
            continue
        else:
           no_of_fut = (prev_no*prev_delta)/current_delta
           if no_of_fut < prev_no:
               verdict.append('Sell')
           else:
               verdict.append('Buy')
           number_to_hedge.append(abs(prev_no - no_of_fut))
           prev_delta = current_delta
           num_fut_list.append(no_of_fut)
           days_to_fut_expiry-=1

    futures["Number_of_Futures"] = num_fut_list
    futures["Buy_Or_Sell"] = verdict
    futures["Futures_to_Buy_or_Sell"] = number_to_hedge
    print(futures)