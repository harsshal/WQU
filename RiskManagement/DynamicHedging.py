__author__ = 'harsshal'

# Get 2 years of data from yahoo for DJIA
# Calculate sigma / volatility from the data
# Lets assume we start trading on Sep 1st, 2014 with contract expiring in Dec.
# Calculate delta from sigma with t = 0.25
# Every day, t will reduce so that in 90 days it will go to 0
# Lets assume that we rollover to March contract on Dec 1st.
# Based on delta, we will be hedging put options with 200*delta contracts long.
# We will calculate total PNL of each quarter based on the closing price of each quarter