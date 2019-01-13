# LSTM for international airline passengers problem with regression framing
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import sys
sys.path.insert(0, '..')
from Indicators import Indicators as ind
# convert an array of values into a dataset matrix

# our data #
features = [ind.RSI, ind.ADL, ind.CCI, ind.momentum_indicator,
            ind.AccumDest, ind.VWAP]    # ind.ADX, ind.MFI, ind.Stochastic, ind.BollingerBands -
                                                                    # need to be added after fix
Google = ind.Stock("googl", None, features)
Google.load_data()

orig_dataset = Google.get_features_as_list(False, False, True)

# till we fix indicators
datasetLen = 3000
dataset = orig_dataset[0][-datasetLen:]
dataset = np.reshape(dataset, (dataset.shape[0], 1))

lable = Google.data[-datasetLen:, :][:, 0]
lable = np.reshape(lable, (lable.shape[0], 1))
# end our data #


def create_dataset(data, lable, look_back=1):
    new_data = []
    new_lable = []

    for i in range(len(data)-look_back-1):
        new_data.append(data[i:(i+look_back), 0])
        new_lable.append(lable[i + look_back, 0])

    return np.array(new_data), np.array(new_lable)


# fix random seed for reproducibility
np.random.seed(7)

# load the dataset
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
train_lable, test_lable= lable[0:train_size, :], dataset[train_size:len(dataset), :]

# reshape into X=t and Y=t+1
look_back = 7
trainX, trainY = create_dataset(train, train_lable, look_back)
testX, testY = create_dataset(test, test_lable, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
