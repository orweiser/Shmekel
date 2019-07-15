from keras import Input
from keras.layers import LSTM as KerasLSTM
from keras.layers import Dense, Reshape, Dropout  # , RNN
from api.core import Model
import numpy as np
import random


class NoneSeriealNet(Model):
    layers: list
    num_of_layers: int
    num_of_rnn_layers: int
    output_activation: str
    _input_shape: tuple
    _output_shape: tuple
    dropout: bool
    dropout_rate: float

    def init(self, layers=[], num_of_layers=0, num_of_rnn_layers=0, input_shape=(10, 1000), output_shape=(3,),
             output_activation='softmax',
             dropout=True, dropout_rate=0.2):
        self.hidden_layers = layers
        self.num_of_layers = num_of_layers
        self.num_of_rnn_layers = num_of_rnn_layers
        self.output_activation = output_activation
        self._input_shape = input_shape
        self._output_shape = output_shape
        self.dropout = dropout
        self.dropout_rate = dropout_rate

    def get_input_output_tensors(self):
        input_shape = self._input_shape
        output_shape = self._output_shape

        input_layer = Input(input_shape)

        # will make more complex cell, might be interesting but not what we meant
        # model = None
        # for layer in self.layers:
        #     if layer['type'] == 'KerasLSTM':
        #       cell = KerasLSTM(layer['size'])(cell)
        #     elif layer['type'] == 'Dense':
        #         cell = Dense(layer['size'], activation=layer['activation_type'])(cell)
        #
        # x = RNN(cell)(input_layer)

        x = input_layer
        for layer in self.hidden_layers:
            if layer['type'] == 'KerasLSTM':
                if self.num_of_rnn_layers > 1:
                    x = KerasLSTM(layer['size'], return_sequences=True)(x)
                    self.num_of_rnn_layers -= 1
                else:
                    x = KerasLSTM(layer['size'], return_sequences=False)(x)
            elif layer['type'] == 'Dense':
                x = Dense(layer['size'], activation=layer['activation_function'])(x)
            if self.dropout:
                x = Dropout(self.dropout_rate)(x)

        x = Dense(np.prod(output_shape), activation=self.output_activation)(x)

        if len(output_shape) > 1:
            x = Reshape(output_shape)(x)
        return input_layer, x

    def __str__(self):
        return 'LSTM_compose'

    def get_default_config(self) -> dict:
        return dict(units=128, input_shape=(10, 1000), output_shape=(3,), output_activation='softmax')
