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

def ticker_data(ticker, start_date):
    """Gets ticker data for the passed ticker within the specified time period"""
    data = si.get_data(ticker, start_date = start_date)['adjclose']
    return data


def ticker_df(tickers, start_date):
    """Creates a dataframe with the data from all the passed tickers
    using the ticker_data function"""
    df = pd.DataFrame()
    for ticker in tickers:
        df[ticker] = ticker_data(ticker, start_date)
        
    return df


def dataframe_cleaner(dataframe):
    """Removes tickers with data less than the specificed date"""
    for ticker in dataframe.columns:
        if True in list(dataframe[ticker].isnull()):
            dataframe.pop(ticker)

            
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


def stationary_tickers(price_dataframe, alpha):
    """Passes ticker data through adf test and if the
        p-value is less than the alpha we add the ticker
        and its halflife to a dictionary"""
    temp_dict = dict()
    
    for ticker in price_dataframe.columns:
        data = price_dataframe[ticker][:-100]
        adf = adfuller(data)[1]
        
        if adf < alpha:
            temp_dict[ticker] = halflife(data)
            
    return temp_dict


def trade_indicators(tickers_dict, price_dataframe):
    """Check all stationary tickers to see if their zscore just
        crossed the two standard deviation threshold"""
    for ticker in tickers_dict.keys():
        hl = tickers_dict[ticker]
        data = price_dataframe[ticker][-hl:]
        
        zscore = (data - data.mean()) / np.std(data)
        
        if zscore[-1] > 2 and zscore[-2] < 2:
            print("Short {}".format(ticker))

        if zscore[-1] < -2 and zscore[-2] > -2:
            print("Long {}".format(ticker))           
