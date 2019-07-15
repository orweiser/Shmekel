from keras import Input
from keras.layers import LSTM as KerasLSTM
from keras.layers import Dense, Reshape, Dropout, concatenate  # , RNN
from api.core import Model
import numpy as np
from layer_in_none_sequential import NoneSequentialLayer
import random


class NoneSequentialArchitecture(Model):
    my_layers: list
    num_of_layers: int
    num_of_rnn_layers: int
    output_activation: str
    _input_shape: tuple
    _output_shape: tuple
    dropout: bool
    dropout_rate: float

    def init(self, num_of_layers=None, num_of_rnn_layers=None, layers=None, output_layer=None, input_shape=None,
             output_shape=None, output_activation=None):
        self.num_of_layers = num_of_layers
        self.num_of_rnn_layers = num_of_rnn_layers
        self.my_layers = []
        for layer in layers:
            self.my_layers.append(NoneSequentialLayer(layer_dict=layer))

        self._input_shape = input_shape
        self._output_shape = output_shape

        self.output_layer = NoneSequentialLayer(layer_dict=output_layer)
        self.output_activation = output_activation;

    def get_input_output_tensors(self):
        input_shape = self._input_shape
        output_shape = self._output_shape

        input_layer = Input(input_shape)

        x = []
        historical2index = {0: 0}  # input index is always 0
        x.append(input_layer)
        for index, layer in enumerate(self.my_layers):
            historical2index[layer.historical_number] = index + 1
            connections = [historical2index[i] for i in layer.connections]
            connections = [x[i] for i in connections]
            if layer.type == 'KerasLSTM':
                if self.num_of_rnn_layers > 1:
                    if len(connections) > 1:
                        x.append(KerasLSTM(layer.size, return_sequences=True)(concatenate(connections)))
                    else:
                        x.append(KerasLSTM(layer.size, return_sequences=True)(connections[0]))
                    self.num_of_rnn_layers -= 1
                else:
                    if len(connections) > 1:
                        x.append(KerasLSTM(layer.size, return_sequences=False)(concatenate(connections)))
                    else:
                        x.append(KerasLSTM(layer.size, return_sequences=False)(connections[0]))
            elif layer.type == 'Dense':
                if len(connections) > 1:
                    x.append(Dense(layer.size, activation=layer.activation_function)(concatenate(connections)))
                else:
                    x.append(Dense(layer.size, activation=layer.activation_function)(connections[0]))
                # if self.dropout:
                #    x = Dropout(self.dropout_rate)(x)

        connections = [historical2index[i] for i in self.output_layer.connections]
        connections = [x[i] for i in connections]
        if len(connections) > 1:
            output = Dense(np.prod(output_shape), activation=self.output_activation)(concatenate(connections))
        else:
            output = Dense(np.prod(output_shape), activation=self.output_activation)(connections[0])

        if len(output_shape) > 1:
            output = Reshape(output_shape)(output)
        return input_layer, output

    def __str__(self):
        return 'LSTM_compose'

    def get_default_config(self) -> dict:
        return dict(units=128, input_shape=(10, 1000), output_shape=(3,), output_activation='softmax')
