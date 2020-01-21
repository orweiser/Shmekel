from keras import Input
from keras.layers import LSTM as KerasLSTM
from keras.layers import Dense, Reshape
from api.core import Model
import numpy as np
import random


class LSTM_compose(Model):
    output_activation: str
    _input_shape: tuple
    _output_shape: tuple
    units: int

    def init(self, units=16, input_shape=(10, 1000), output_shape=(3,), output_activation='softmax'):
        self.output_activation = output_activation
        self._input_shape = input_shape
        self._output_shape = output_shape
        self.units = units

    def get_input_output_tensors(self):
        input_shape = self._input_shape
        output_shape = self._output_shape

        input_layer = Input(input_shape)
        x = input_layer

        dense_activation_types = ['relu', 'sigmoid', 'tanh']
        layer_types = ['KerasLSTM', 'Dense']
        for i in range(4):
            layer_type = random.choice(layer_types)
            if layer_type == 'KerasLSTM':
                x = KerasLSTM(self.units, return_sequences=True)(x)
            elif layer_type == 'Dense':
                x = Dense(4, activation=random.choice(dense_activation_types))(x)

        x = KerasLSTM(self.units)(x)
        x = Dense(np.prod(output_shape), activation=self.output_activation)(x)

        if len(output_shape) > 1:
            x = Reshape(output_shape)(x)
        return input_layer, x

    def __str__(self):
        return 'LSTM_compose'

    def get_default_config(self) -> dict:
        return dict(units=128, input_shape=(10, 1000), output_shape=(3,), output_activation='softmax')

