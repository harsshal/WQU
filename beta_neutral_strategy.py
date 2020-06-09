#import scipy as sp
#from numpy import *
#import scipy.stats as ss
#delay=DELAY
##dr is the Data Registry
#valid = dr.GetData("UNIVID")
#Lines above will be added as a common header to your code, without "#".
# -------------------------Write your code below------------------------

close = dr.GetData("close")
open = dr.GetData("Open")
sector = dr.GetData("sector")
returns = dr.GetData("returns")
cap = dr.GetData("cap")

@np.vectorize
def selected(elmt,sec): return elmt == sec  # Or: selected = numpy.vectorize(wanted_set.__contains__)

def Generate(di,alpha):
  #selecting 90 day returns
  ret90 = returns[di-delay-89:di-delay+1, :]
  
  #replacing NANs with 0's
  ret90 = where(ret90 != ret90, 0, ret90)
  #selecting 90 day cap to be used as weights
  cap90 = cap[di-delay-89:di-delay+1, :]
  
  #replacing NANs with 0's
  cap90 = where(cap90 != cap90, 0, cap90)
  
  #calculating market returns for 90 days, using cap as weights
  ret90_market = average(ret90, axis=1, weights=cap90)
  
  #calculating covariance of market returns with individual stock returns
  ret90 -= mean(ret90, axis=0)
  ret90_market -= mean(ret90_market)
  covar = dot(ret90_market, ret90)
  
  #calculating beta
  beta = covar / var(ret90_market)
  alpha[:] = -beta
  alpha[:] = where(valid[di,:],alpha[:],nan) # valid check
  #Python code is copyright © 2001-2014 Python Software Foundation. All Rights Reserved
  
  sec_list = sector.unique()
  for sec in sec_list:
    centroid = mean(ret[selected(sector[di-delay, :],sec)])
    alpha[:] = ret[di-delay,:] - centraoid
