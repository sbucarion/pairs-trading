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
import warnings


def ticker_data(ticker_list, start_date, end_date):
    #Generate dates for stationarity and trading for dataframe
    indexes = get_data('AAPL', start_date, end_date).index
    df = pd.DataFrame(index = indexes)
    
    #Input historical data for each ticker into dataframe
    for ticker in ticker_list:
        try:
            df[ticker] = get_data(ticker, start_date = start_date, end_date = end_date)['adjclose']
            
        except Exception as e:
            print(ticker)
            print(e)

            
    #Remove tickers that did not pull data correctly        
    for ticker in df:
        if sum(df[ticker]) > 0:
            continue
            
        else:
            df.pop(ticker)
        
    return df


#Remove non stationary tickers from the dataframes
def stationarity(training_df, trading_df, alpha):
    for ticker in training_df:
        adf = adfuller(training_df[ticker])[1] #Test to check for stationarity
        
        if adf > alpha:
            training_df.pop(ticker)
            trading_df.pop(ticker) 


#Used to create lookback length for trading period
def halflife(spread):
    """Regression on the pairs spread to find lookback
        period for trading"""
    x_lag = np.roll(spread,1)
    x_lag[0] = 0
    y_ret = spread - x_lag
    y_ret[0] = 0
    
    x_lag_constant = sm.add_constant(x_lag)
    
    res = sm.OLS(y_ret,x_lag_constant).fit()
    halflife = -np.log(2) / res.params[1]
    halflife = int(round(halflife))
    return halflife


#Create trading spread for each stationary tickers
def ticker_spread(training_data, testing_data):
    hl = halflife(training_data)   
    spread = testing_data.iloc[-hl:]   
    
    spread = (spread - spread.mean()) / np.std(spread)
    
    return spread


#Generate trade signals if the zscore of spread crosses +- 2 standard deviations
def trade_signals(training_data, testing_data):   
    for ticker in testing_data:
        spread = ticker_spread(training_data[ticker], testing_data[ticker])
        
        if spread[-1] > 2 and spread[-2] < 2:
            print("Short", ticker)
            
        if spread[-1] < -2 and spread[-2] > -2:
            print("Long", ticker)

            
#Wrapper Function to store all algorithm processes
def trades(start_date, end_date, tickers, alpha):
    df = ticker_data(tickers, sd, ed)
    coint_df = df.iloc[:-100]
    trading_df = df.iloc[-100:]

    stationarity(coint_df, trading_df, alpha)

    trade_signals(coint_df, trading_df)         

    
start_date = date.today() - timedelta(days = 67, weeks = 260)
end_date = date.today() + timedelta(days = 1)

alpha = 0.01

tickers = si.tickers_sp500()

trades(start_date, end_date, tickers, alpha)
