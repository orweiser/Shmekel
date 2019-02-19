from rnn import Generate_Data as gd
import numpy as np
from keras.layers import Input, LSTM, Dense, TimeDistributed, concatenate
from keras.models import Model


# for i in range(3):
#     data = gd.generate_data(i+1, 0.01*i, 1000)
#     plt.plot(data)
# plt.show()

data = np.array(gd.generate_data(0, 0, length=10000))

upThreshold = 1.008
downThreshold = 0.992
time_batch = 64
labels = []
finalData = []

yesterday = data[0]
for today in data[1:]:
    if today >= upThreshold*yesterday:
        labels.append(1)
    elif today <= downThreshold*yesterday:
        labels.append(-1)
    else:
        labels.append(0)
    yesterday = today

for i in range(len(data) - time_batch):    # without the last data point
    finalData.append(np.reshape(data[i:i+time_batch], (time_batch, 1)))
finalLabels = labels[-len(finalData):]
finalData = np.stack(finalData)
input_shape = [time_batch, 1]
model_input = Input(shape=input_shape, dtype='float', name='model_input')
rnn = LSTM(units=10, return_sequences=False)(model_input)
model_output = Dense(1, activation='tanh', name='model_output')(rnn)

model = Model(inputs=model_input, outputs=model_output)
model.compile(optimizer='adam', loss='mse', metrics=['acc'])

model.summary()

model.fit(finalData, finalLabels, validation_split=0.1, verbose=1, epochs=10)



# print(labels)
# print(data)


