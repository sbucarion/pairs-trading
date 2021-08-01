# pairs-trading
Generate trading signals on stationary stock pairs

#Single Stock Stat Arb: 
Find stationary stocks using the past 5 years of data. Trade the stationary stocks when their zscore rises or falls two standard deviations.
Zscore length is calculated using the Ornstein-Uhlenbeck process to find the half life of a mean reverting time series

#Pairs Stat Arb:
Same principle as the single stock but instead we find two stocks that have a spread that is stationary.
The spread is Stock A - (Stock B * Ratio) -> The ratio is found by running a linear regression on the pairs data
If the spread is stationary then we conclude the pairs are cointegrated
Then we follow the same steps as single stock stat arb; however, instead of stock data we use the spread data
