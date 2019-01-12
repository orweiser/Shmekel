import sys, os
import numpy as np
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
d_test = 100
d_train = 2900
n_test = np.zeros((d_test, len(x)))
n_train = np.zeros((d_train, len(x)))
for i in range(len(x)):
    n_train[:, i] = x[i][-(d_train + d_test):-d_test]
    n_test[:, i] = x[i][-d_test:]
##
input_size = n_train.shape[0]
output_size = 5  # candle size
layer1_size = 256
gru_size = 256

model = Sequential()
model.add(Dense(layer1_size, activation='relu', input_dim=input_size))
# model.add(Dense(5, activation='sigmoid'))
model.summary()
model.add(layers.Flatten())
model.summary()
model.add(GRU(input_dim=layer1_size, output_dim=gru_size, activation='tanh', return_sequences=False))
model.add(Dense(output_size, activation='relu', input_dim=gru_size))
model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy'])

print('Start training...')
model.fit(n_train, Google.data[-(d_train+d_test):-d_test, :], epochs=10, batch_size=32)
print('Done training!')