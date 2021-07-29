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
            

def ticker_pairs(tickers):
    """Create pairs of the tickers in the dataframe"""
    pairs = []
    for ticker1 in tickers:
        for ticker2 in tickers:
            if (ticker1 != ticker2) and ((ticker2 + "/" + ticker1) not in pairs):
                pair = ticker1+"/"+ticker2
                pairs.append(pair)
                               
    return pairs


@jit(nopython = True)
def create_pairs(df_columns):
    """Create pairs of all the tickers in the dataframe.
        Use jit to speed up the nested for loop"""
    pairs = []
    for ticker in df_columns:
        for ticker2 in df_columns:
            if ticker != ticker2: #and ((ticker2 + "/" + ticker) not in pairs):
                pair = ticker + "/" + ticker2
                pairs.append(pair)
                
    return pairs


def hedge_ratio(a, a_data, b, b_data):
    """Linear Regression, a, first ticker in pair, is x
        b is the second ticker, y. X is IV and y is DV. 
        The ratio is used the create the spread for the ticker"""
    a_constant = sm.add_constant(a_data)
    
    #Run linear regression on the pairs data for ratio
    results = sm.OLS(b_data,a_constant).fit()
    ratio = results.params[[a][0]] 
    
    return ratio


def check_stationarity(a, b, a_data, b_data, alpha):
    """Pass in the pairs price data and function will output a 
        dictionary if the p-value is less than the alpha for the 
        adf test which tests for stationarity"""
    ratio = hedge_ratio(a, a_data, b, b_data)
    spread = b_data - (ratio * a_data)
    adf = adfuller(spread)[1]
    
    if adf <= alpha:
        return {'adfuller': adf, 'ratio': ratio, 'spread': spread}
    
    else:
        return {'adfuller': -1, 'ratio': -1, 'spread': -1}

    
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


def pair_features(pairs, price_dataframe, alpha):
    """Pass in pairs to get the p-value, ratio, spread, correlation,
        and halflife for each pair in the pairs list and add that data 
        to a pairs dictionary"""
    temp_dict = dict()
    
    for pair in pairs:
        a,b = pair.split("/")
        a_data = price_dataframe[a][:-100] 
        b_data = price_dataframe[b][:-100] 

        correl = pearsonr(a_data,b_data)[0]
        
        if (correl > 0.9 or correl < -0.9): 
            adf_dict = check_stationarity(a, b, a_data, b_data, alpha)
            
            if adf_dict['adfuller'] != -1:
                adf_dict['correl'] = round(correl, 3)
                
                adf_dict['halflife'] = halflife(list(adf_dict['spread']))
                
                temp_dict[pair] = adf_dict
                
        
       
    return temp_dict


def trading_zscore(pair, price_dataframe, ticker_dictionary):
    """Takes in the pair dictionary and creates a zscore that will
        be used for creating trade signals. The length of the zscore is equal 
        to the halflife found from the stationary data"""
    features = ticker_dictionary[pair]
    
    ratio = features['ratio']
    hl = features['halflife']
    
    a,b = pair.split("/")
    
    a_data = price_dataframe[a][-hl:]
    b_data = price_dataframe[b][-hl:]
    
    spread = b_data - (a_data * ratio)
    
    zscore = ((spread - spread.mean()) / np.std(spread))
     
    return zscore
    
    
def trade_indicator(price_dataframe, ticker_dictionary):
    """Checks each pair in the dictionary and if the zscore
        has crossed a specificed threshold (2 Standard Deviations)
        then a trade signal will be printed"""
    for pair in ticker_dictionary.keys():
        a, b = pair.split("/")
        
        ratio = ticker_dictionary[pair]['ratio']
        zscore = trading_zscore(pair, price_dataframe, ticker_dictionary)
        
        
        if ratio > 0:
            if zscore[-1] > 2 and zscore[-2] < 2:
                print('Buy {} shares of {} and sell 1 share of {}'.format(abs(round(ratio,2)), a, b))
                
                
            if zscore[-1] < -2 and zscore[-2] > -2:
                print('Sell {} shares of {} and buy 1 share of {}'.format(abs(round(ratio,2)), a, b))
                
                
        if ratio < 0:
            if zscore[-1] > 2 and zscore[-2] < 2:
                print('Sell {} shares of {} and sell 1 share of {}'.format(abs(round(ratio,2)), a, b))
                
                
            if zscore[-1] < -2 and zscore[-2] > -2:
                print('Buy {} shares of {} and buy 1 share of {}'.format(abs(round(ratio,2)), a, b))
