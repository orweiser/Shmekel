import sys, os
import numpy as np
import sys
sys.path.insert(0, '..')
from Indicators import Indicators as ind
import keras as K
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Activation, GRU

features = [ind.RSI, ind.ADL, ind.CCI, ind.momentum_indicator,
            ind.AccumDest, ind.VWAP]    # ind.ADX, ind.MFI, ind.Stochastic, ind.BollingerBands -
                                                                    # need to be added after fix
Google = ind.Stock("googl", None, features)
Google.load_data()
# print(Google.data)
# print(Google.get_features_as_list(False, False, True))

x = Google.get_features_as_list(False, False, True)

# till we fix indicators
d_train = 2900
d_test = 100
x_train = np.zeros((d_train, len(x)))
x_test = np.zeros((d_test, len(x)))

for i in range(len(x)):
    x_train[:, i] = x[i][-(d_train + d_test):-d_test]
    x_test[:, i] = x[i][-d_test:]
##
input_size = x_train.shape[1]
output_size = 5  # candle size
layer1_size = 64  # 256
gru_size = 256

# x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
# x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

model = Sequential()
model.add(Dense(units=layer1_size, activation='relu'))
model.add(GRU(units=10, activation='tanh', recurrent_activation='hard_sigmoid', return_sequences=True))
model.add(GRU(units=10, activation='tanh', recurrent_activation='hard_sigmoid', return_sequences=False))
#model.add(Dense(output_size, activation='relu'))

model.build((None, d_train, output_size))
model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy'])
model.summary()
print('Start training...')
model.fit(x_train, Google.data[-(d_train+d_test):-d_test, :], epochs=100, batch_size=32)
print('Done training!')