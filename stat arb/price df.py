import numpy as np
import pandas as pd
from yahoo_fin import options
from yahoo_fin import stock_info as si
from yahoo_fin.stock_info import *
from yahoo_fin import *
import math
from statsmodels.tsa.stattools import adfuller
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm
from scipy import stats
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from numpy.random import randn
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LinearRegression
from datetime import datetime as dt
from datetime import timedelta
import datetime
from datetime import date
import threading
from threading import Thread
import time
import numba
from numba import jit
from numba.typed import List
import df_functions as dff
import warnings

#Remove VSCode Future Warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#Call all tickers in sp500
tickers = si.tickers_sp500()

#Create date roughly 5 years ago to collect stationary data
today = date.today()
start_date = today - timedelta(days = 1970)

#Collect price data for all the tickers 
df = dff.ticker_df(tickers, start_date)

#Remove tickers with less data than the start date
dff.dataframe_cleaner(df)

#Create pairs
df_tickers = List(df.columns)
pairs = dff.create_pairs(df_tickers)

#Create dictionary with stationary pairs and their data
pairs_dict = dff.pair_features(pairs, df, 0.01)

#Create trade signals for stationary pairs zscore crossing the threshold
dff.trade_indicator(df, pairs_dict)