from keras import Input
from keras.layers import LSTM as KerasLSTM
from keras.layers import Dense, Reshape, Dropout
from api.core import Model
import numpy as np


class LSTM_DropOut(Model):
    output_activation: str
    _input_shape: tuple
    _output_shape: tuple
    units: int

    def init(self, units=128, input_shape=(10, 1000), output_shape=(3,), output_activation='softmax'):
        self.output_activation = output_activation
        self._input_shape = input_shape
        self._output_shape = output_shape
        self.units = units

    def get_input_output_tensors(self):
        input_shape = self._input_shape
        output_shape = self._output_shape

        input_layer = Input(input_shape)

        x = KerasLSTM(self.units)(input_layer)
        x = Dropout(0.2)(x)
        x = KerasLSTM(self.units)(x)
        x = Dropout(0.2)(x)
        x = KerasLSTM(self.units)(x)
        x = Dropout(0.2)(x)
        x = KerasLSTM(self.units)(x)
        x = Dense(np.prod(output_shape), activation=self.output_activation)(x)

        if len(output_shape) > 1:
            x = Reshape(output_shape)(x)
        return input_layer, x

    def __str__(self):
        return 'LSTM_DropOut-units_' + str(self.units)

    def get_default_config(self) -> dict:
        return dict(units=128, input_shape=(10, 1000), output_shape=(3,), output_activation='softmax')

