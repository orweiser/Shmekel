from keras.layers import Input, LSTM, Dense, concatenate
from keras.models import Model
import numpy as np
from keras.utils.vis_utils import plot_model

from Indicators import Indicators as Ind


features = [Ind.RSI, Ind.ADL, Ind.CCI, Ind.momentum_indicator, Ind.AccumDest, Ind.VWAP]
input_len = len(features)
Google = Ind.Stock("googl", None, features)
Google.load_data()
data = np.zeros((3000, input_len, 1))
temp = Google.get_features_as_list(False, False, True)
for i, feature in enumerate(temp):
    data[:,i,:] = feature[-3000:].reshape((3000,1))
labels = []
yesterday = Google.data[0][3]
for i, today in enumerate(Google.data[1:]):
    if today[3] > yesterday * 1.1:
        labels.append(1)
    elif today[3] < yesterday * 0.9:
        labels.append(-1)
    else:
        labels.append(0)
    yesterday = today[3]


daily_input = Input(shape=(input_len, 1), dtype='float', name='daily_input')
# weekly_input = Input(shape=(input_len, 1), dtype='float', name='weekly_input')

xd = Dense(units=64, activation='relu')(daily_input)
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
              loss='mse')

model.summary()

model.fit(data, labels[-3000:], validation_split=0.1)

# plot_model(model, to_file='model.png')
