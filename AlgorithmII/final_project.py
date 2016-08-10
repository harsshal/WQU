# Get current constituents of SPX
#
# Get data for 10 years for those constituents
#
# Get constituents of DJIA of past 10 years
#
# Calculate daily returns of all of the above
#
# Design function port(M,X,N,P)
#
# Rank trend lines on moving average M of year for every year,
#
# in 2 groups SPX and that year's DJIA
#
# We will be adding X units of cash for N years / investing X*N amount each year
#
# After N years, we will worst performing P% stocks
#
# and replace with highest potential trending lines stocks
#
# Pass this function to optimizer and get optimal values for 4 parameters

# SPX,A,AA,AAL,AAP,AAPL,ABBV,ABC,ABT,ACN,ADBE,ADI,ADM,ADP,ADS,ADSK,ADT,AEE,AEP,AES,AET,AFL,AGN,AIG,AIV,AIZ,AKAM,ALL,ALLE,ALXN,AMAT,AME,AMG,AMGN,AMP,AMT,AMZN,AN,ANTM,AON,APA,APC,APD,APH,ARG,ATVI,AVB,AVGO,AVY,AXP,AZO,BA,BAC,BAX,BBBY,BBT,BBY,BCR,BDX,BEN,BF/B,BHI,BIIB,BK,BLK,BLL,BMY,BRCM,BRK/B,BSX,BWA,BXLT,BXP,C,CA,CAG,CAH,CAM,CAT,CB,CBG,CBS,CCE,CCI,CCL,CELG,CERN,CF,CHD,CHK,CHRW,CI,CINF,CL,CLX,CMA,CMCSA,CME,CMG,CMI,CMS,CNP,CNX,COF,COG,COH,COL,COP,COST,CPB,CPGX,CRM,CSCO,CSRA,CSX,CTAS,CTL,CTSH,CTXS,CVC,CVS,CVX,D,DAL,DD,DE,DFS,DG,DGX,DHI,DHR,DIS,DISCA,DISCK,DLPH,DLTR,DNB,DO,DOV,DOW,DPS,DRI,DTE,DUK,DVA,DVN,EA,EBAY,ECL,ED,EFX,EIX,EL,EMC,EMN,EMR,ENDP,EOG,EQIX,EQR,EQT,ES,ESRX,ESS,ESV,ETFC,ETN,ETR,EW,EXC,EXPD,EXPE,F,FAST,FB,FCX,FDX,FE,FFIV,FIS,FISV,FITB,FLIR,FLR,FLS,FMC,FOSL,FOX,FOXA,FSLR,FTI,FTR,GAS,GD,GE,GGP,GILD,GIS,GLW,GM,GMCR,GME,GOOG,GOOGL,GPC,GPS,GRMN,GS,GT,GWW,HAL,HAR,HAS,HBAN,HBI,HCA,HCN,HCP,HD,HES,HIG,HOG,HON,HOT,HP,HPE,HPQ,HRB,HRL,HRS,HSIC,HST,HSY,HUM,IBM,ICE,IFF,ILMN,INTC,INTU,IP,IPG,IR,IRM,ISRG,ITW,IVZ,JBHT,JCI,JEC,JNJ,JNPR,JPM,JWN,K,KEY,KHC,KIM,KLAC,KMB,KMI,KMX,KO,KORS,KR,KSS,KSU,L,LB,LEG,LEN,LH,LLL,LLTC,LLY,LM,LMT,LNC,LOW,LRCX,LUK,LUV,LVLT,LYB,M,MA,MAC,MAR,MAS,MAT,MCD,MCHP,MCK,MCO,MDLZ,MDT,MET,MHK,MJN,MKC,MLM,MMC,MMM,MNK,MNST,MO,MON,MOS,MPC,MRK,MRO,MS,MSFT,MSI,MTB,MU,MUR,MYL,NAVI,NBL,NDAQ,NEE,NEM,NFLX,NFX,NI,NKE,NLSN,NOC,NOV,NRG,NSC,NTAP,NTRS,NUE,NVDA,NWL,NWS,NWSA,O,OI,OKE,OMC,ORCL,ORLY,OXY,PAYX,PBCT,PBI,PCAR,PCG,PCL,PCLN,PCP,PDCO,PEG,PEP,PFE,PFG,PG,PGR,PH,PHM,PKI,PLD,PM,PNC,PNR,PNW,POM,PPG,PPL,PRGO,PRU,PSA,PSX,PVH,PWR,PX,PXD,PYPL,QCOM,QRVO,R,RAI,RCL,REGN,RF,RHI,RHT,RIG,RL,ROK,ROP,ROST,RRC,RSG,RTN,SBUX,SCG,SCHW,SE,SEE,SHW,SIG,SJM,SLB,SLG,SNA,SNDK,SNI,SO,SPG,SPGI,SPLS,SRCL,SRE,STI,STJ,STT,STX,STZ,SWK,SWKS,SWN,SYF,SYK,SYMC,SYY,T,TAP,TDC,TE,TEL,TGNA,TGT,THC,TIF,TJX,TMK,TMO,TRIP,TROW,TRV,TSCO,TSN,TSO,TSS,TWC,TWX,TXN,TXT,TYC,UA,UAL,UHS,UNH,UNM,UNP,UPS,URBN,URI,USB,UTX,V,VAR,VFC,VIAB,VLO,VMC,VNO,VRSK,VRSN,VRTX,VTR,VZ,WAT,WBA,WDC,WEC,WFC,WFM,WHR,WM,WMB,WMT,WRK,WU,WY,WYN,WYNN,XEC,XEL,XL,XLNX,XOM,XRAY,XRX,XYL,YHOO,YUM,ZBH,ZION,ZTS

import pandas as pd

def portfolio(close, M,X,N,P):
    ma = close.rolling(M).mean()
    print(ma)

    ma[ma.index.year==2014]

def get_yahoo_data(tickers,start,end,only_close=1):
    import pandas_datareader.data as web

    panel = web.DataReader(tickers, 'yahoo', start, end)

    if only_close !=1:
        return panel
    else:
        return panel['Adj Close']

def main():
    gspc = pd.read_csv('index_constituents', header=None, index_col=0, nrows=1)
    #print(gspc)

    close = get_yahoo_data(gspc.loc['GSPC_current'].tolist(), '20140101', '20160101')
    #print(close.head())

    portfolio(close,10,10,2,3)


if __name__ == '__main__':
    main()