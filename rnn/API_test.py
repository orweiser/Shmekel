from keras.layers import Input, LSTM, Dense, TimeDistributed, concatenate
from keras.models import Model
# from keras.utils.vis_utils import plot_model
#from FeatureSpace.AD import *
#from FeatureSpace.ADL import *
from Indicators import *
from Utils.Data import *

upThreshold = 1.01
downThreshold = 0.99
time_batch = 64

stocks_info = DataReader('../Shmekel_config.txt').get_data_info()
stocks_info = [s for s in stocks_info if s[1] > 3000]
stock_info = stocks_info[0]
feature_list = [High(), Low()]
stock = Stock(stock_tckt=stock_info[0], feature_list=feature_list)

labels = []
finalData = []
cnd = stock.data[0]
yesterday = cnd[0][3]
features_data = np.vstack(stock.numerical_feature_list).transpose()
for i, today in enumerate(cnd[1:]):
    if today[3] > yesterday * upThreshold:
        labels.append(1)
    elif today[3] < yesterday * downThreshold:
        labels.append(-1)
    else:
        labels.append(0)
    yesterday = today[3]
    finalData.append(features_data[i:i + time_batch, :])

finalData = np.stack(finalData[:-(time_batch-2)])
finalLabels = labels[time_batch-2:]

daily_shape = [time_batch, feature_list.__len__()]
daily_input = Input(shape=daily_shape, dtype='float', name='daily_input')
# weekly_input = Input(shape=(input_len, 1), dtype='float', name='weekly_input')

xd = TimeDistributed(Dense(units=64, activation='relu'))(daily_input)
daily_out = LSTM(units=30, return_sequences=False)(xd)
# xw = Dense(units=64, activation='relu')(weekly_input)
# weekly_out = LSTM(units=30, return_sequences=False)(xw)

# x = concatenate([daily_out, weekly_out])
x = Dense(64, activation='relu')(daily_out)

main_output = Dense(1, activation='tanh', name='main_output')(x)

# model = Model(inputs=[daily_input, weekly_input], outputs=main_output)
model = Model(inputs=daily_input, outputs=main_output)
model.compile(optimizer='adam',
              loss='mse', metrics=['acc'])

model.summary()

model.fit(finalData, finalLabels, validation_split=0.1, verbose=1, epochs=10)

# plot_model(model, to_file='model.png')
