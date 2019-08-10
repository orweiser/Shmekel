from keras import Input
from keras.layers import LSTM as KerasLSTM
from keras.layers import Dense, Reshape
from api.core import Model
import numpy as np


class LSTM(Model):
    output_activation: str
    _input_shape: tuple
    _output_shape: tuple
    units: int

    def init(self, units=64, input_shape=(5, 5), output_shape=(2,), output_activation='softmax'):
        self.output_activation = output_activation
        self._input_shape = input_shape
        self._output_shape = output_shape
        self.units = units

    def get_input_output_tensors(self):
        input_shape = self._input_shape
        output_shape = self._output_shape

        input_layer = Input(input_shape)

        x = KerasLSTM(self.units)(input_layer)
        x = Dense(np.prod(output_shape), activation=self.output_activation)(x)

        if len(output_shape) > 1:
            x = Reshape(output_shape)(x)
        return input_layer, x

    def __str__(self):
        return 'LSTM-units_' + str(self.units)

    def get_default_config(self) -> dict:
        return dict(units=64, input_shape=(28, 28), output_shape=(10,), output_activation='softmax')


# def lstm(dataset_name='mnist', depth=1, units=64, base_activation='sigmoid', skip_connections=False,
#                     batch_normalization=False, batch_norm_after_activation=True, batch_norm_before_output=False,
#                     input_shape=None, output_shape=None, output_activation='softmax', name=None):
#     """
#     a function to create a fully connected perceptron model
#     :param dataset_name: optional. used to determine input and output shape when they are not specified
#     :param depth: the number of hidden layers in the model
#     :param width: number of units in each hidden layer
#     :param base_activation: activation for hidden layers
#     :param skip_connections: boolean. whether or not to add skip connectionf between hidden layers
#     :param batch_normalization: boolean
#     :param batch_norm_after_activation: boolean. if true then l->activation->bn otherwise l->bn->activation
#     :param batch_norm_before_output: boolean. use batch norm after last hidden layer
#     :param input_shape: model's input shape. if not specified, determined by dataset_name
#     :param output_shape: model's output shape. if not specified, determined by dataset_name
#     :param output_activation: output layer's activation
#     :param name: model's name
#     :return: a keras model
#     """
#     def add_layer(x, is_last_hidden_layer=False):
#         x = Dense(units)(x)
#         if (not batch_normalization) or (is_last_hidden_layer and not batch_norm_before_output):
#             return Activation(base_activation)(x)
#
#         if not batch_norm_after_activation:
#             x = BatchNormalization()(x)
#             x = Activation(base_activation)(x)
#         else:
#             x = Activation(base_activation)(x)
#             x = BatchNormalization()(x)
#
#         return x
#     print("LSTM!")
#     input_shape = input_shape or {'mnist': (28, 28)}[dataset_name]
#     output_shape = output_shape or {'mnist': (10,)}[dataset_name]
#
#     input_layer = Input(input_shape)
#
#     x = LSTM(units)(input_layer)
#     x = Dense(10, activation=output_activation)(x)
#     print("len(output_shape): ", len(output_shape))
#     if len(output_shape) > 1:
#         x = Reshape(output_shape)(x)
#
#     return dict(inputs=input_layer, outputs=x)
