from keras.layers import Input, LSTM, Dense, concatenate
from keras.models import Model
from keras.utils.vis_utils import plot_model




input_len = 10

daily_input = Input(shape=(input_len, 1), dtype='float', name='daily_input')
weekly_input = Input(shape=(input_len, 1), dtype='float', name='weekly_input')

xd = Dense(units=64, activation='relu')(daily_input)
daily_out = LSTM(units=30, return_sequences=False)(xd)
xw = Dense(units=64, activation='relu')(weekly_input)
weekly_out = LSTM(units=30, return_sequences=False)(xw)

x = concatenate([daily_out, weekly_out])
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

main_output = Dense(1, activation='sigmoid', name='main_output')(x)

model = Model(inputs=[daily_input, weekly_input], outputs=main_output)
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              loss_weights=[1])

model.summary()

model.fit(data, labels)


#plot_model(model, to_file='model.png')
