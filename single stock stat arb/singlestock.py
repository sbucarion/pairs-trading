import numpy as np
import pandas as pd
from yahoo_fin import stock_info as si
from yahoo_fin.stock_info import *
from yahoo_fin import *
import math
from statsmodels.tsa.stattools import adfuller
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression
from datetime import datetime as dt
from datetime import timedelta
import datetime
from datetime import date
import time
import singlestock_functions as ssf
import warnings

#Remove VSCode Future Warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#Call all tickers in sp500
tickers = si.tickers_sp500()

#Create date roughly 5 years ago to collect stationary data
today = date.today()
start_date = today - timedelta(days = 1970)

#Collect price data for all the tickers 
df = ssf.ticker_df(tickers, start_date)

#Remove tickers with less data than the start date
ssf.dataframe_cleaner(df)

#Find stationary tickers
tickers = ssf.stationary_tickers(df, 0.01)

#Indicate which ticker zscore has recently just crossed the threshold
ssf.trade_indicators(tickers, df)