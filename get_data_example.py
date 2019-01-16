from data import get_data_info
import numpy as np


all_stocks_info = get_data_info()
# stock_info is now a list of tuples as:
# (stock_name, number_of_samples, path of the file)

# we can now use the info to load a random stock with more than 3000 samples:
#   first, lets discard all the stocks with less than 3000 samples:
all_stocks_info = [s for s in all_stocks_info if s[1] > 3000]
print('dropped all the stocks with less than 3000 samples. left with', len(all_stocks_info), 'stocks.')

#   now lets choose a stock at random
stock_info = all_stocks_info[np.random.randint(len(all_stocks_info))]
print('chose:', stock_info)


# we can now define a stock:
from Indicators import *
stock = Stock(stock_tckt=stock_info[0])

# to access the stock data, just subscript it
numerical_data, dates = stock.data
# data is a 2-D numpy array with the values of Open High Low Close Volume, and dates is a list of dates
print('numerical data shape:', numerical_data.shape, '\ndates length:', len(dates))


# to compute features, we need to define the stock with the features we want

#   for the sake of the example, lets say we want the following features, even though they are degenerate:
feature_list = [High(), Candle(with_volume=True), Candle(with_volume=False)]

stock = Stock(stock_tckt=stock_info[0], feature_list=feature_list)
# or we could have just do the following:
stock.set_feature_list(feature_list)

# either way, stock now holds a list of features, we can access it as follows:
computed_feature_list = stock.numerical_feature_list

# lets see what we got:
for i, f in enumerate(computed_feature_list):
    print('Feature', i, 'shape:', f.shape)
# see that it corresponds to our original feature list

# we can also get it as a matrix:
m = stock.feature_matrix
print('matrix shape:', m.shape)

# but what about the dates?
# currently we only have dates in their not-numerical shape
# still, we can add dates as a non numerical feature:
feature_list = [High(), DateTuple(), Candle(with_volume=True), Candle(with_volume=False)]
stock.set_feature_list(feature_list)

# we still have all the numerical features from before, but now we also have:
not_numerical_feature_list = stock.non_numerical_feature_list
print('there is', len(not_numerical_feature_list), 'non numerical features')
# we only have 1 non numerical feature - date_tuple
date_tuple = not_numerical_feature_list[0]
print('date_tuple length:', len(date_tuple))

