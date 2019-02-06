from keras.layers import Input, LSTM, Dense, TimeDistributed, concatenate
from keras.models import Model
# import numpy as np
# from keras.utils.vis_utils import plot_model
#from FeatureSpace.AD import *
#from FeatureSpace.ADL import *
from Indicators import *
from Utils.Data import *

# weiser your code uses the old data code and requires
stocks_info = DataReader('../Shmekel_config.txt').get_data_info()
print('stock_info:', stocks_info)
stocks_info = [s for s in stocks_info if s[1] > 3000]
print('stock_info:', stocks_info)
stock_info = stocks_info[np.random.randint(len(stocks_info))]
feature_list = [Candle()]
print('stock_info:', stock_info)
stock = Stock(stock_tckt=stock_info[0], feature_list=feature_list)
print('stock:', stock.feature_matrix)

"""
features = [Ind.RSI, Ind.ADL, Ind.CCI, Ind.momentum_indicator, Ind.AccumDest, Ind.VWAP]
input_len = len(features)
Google = Ind.Stock("googl", None, features)
Google.load_data()
data = np.zeros((3000, input_len, 1))
temp = Google.get_features_as_list(False, False, True)
for i, feature in enumerate(temp):
    data[:,i,:] = feature[-3000:].reshape((3000,1))
    """
labels = []
finalData = []
tempData = stock.data[0]
yesterday = tempData[0][3]
for i, today in enumerate(tempData[1:]):
    if today[3] > yesterday * 1.01:
        labels.append(1)
    elif today[3] < yesterday * 0.99:
        labels.append(-1)
    else:
        labels.append(0)
    yesterday = today[3]
    finalData.append(tempData[i:i+7])

finalData = np.stack(finalData[:-5])
daily_input = Input(shape=(7, 5), dtype='float', name='daily_input')
# weekly_input = Input(shape=(input_len, 1), dtype='float', name='weekly_input')

xd = TimeDistributed(Dense(units=64, activation='relu'))(daily_input)
daily_out = LSTM(units=30, return_sequences=False)(xd)
# xw = Dense(units=64, activation='relu')(weekly_input)
# weekly_out = LSTM(units=30, return_sequences=False)(xw)

# x = concatenate([daily_out, weekly_out])
x = Dense(64, activation='relu')(daily_out)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

main_output = Dense(1, activation='tanh', name='main_output')(x)

# model = Model(inputs=[daily_input, weekly_input], outputs=main_output)
model = Model(inputs=daily_input, outputs=main_output)
model.compile(optimizer='rmsprop',
              loss='mse', metrics=['acc'])

model.summary()

model.fit(finalData, labels[:-5], validation_split=0.1, verbose=2, epochs=10)

# plot_model(model, to_file='model.png')
