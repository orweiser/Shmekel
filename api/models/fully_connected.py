from ..core.model import *
from keras.layers import Dense, Flatten, Activation, BatchNormalization, Add, Reshape
from keras import Input
from keras.regularizers import l1_l2
import numpy as np


class FullyConnected(Model):
    # define types to prevent IDE warnings
    output_activation: str
    batch_norm_before_output: bool
    batch_norm_after_activation: bool
    batch_normalization: bool
    skip_connections: bool
    base_activation: str
    width: int
    depth: int
    name: str
    _input_shape: tuple
    _output_shape: tuple
    regularization_value : float
    weights_path: str

    def init(self, depth=1, width=32, base_activation='relu', skip_connections=False, batch_normalization=False,
             batch_norm_after_activation=True, batch_norm_before_output=False, input_shape=(28, 28),
             output_shape=(10,), output_activation='softmax', name=None, weights_path=None):
        self.depth = depth
        self.depth = depth
        self.width = width
        self.base_activation = base_activation
        self.skip_connections = skip_connections
        self.batch_normalization = batch_normalization
        self.batch_norm_after_activation = batch_norm_after_activation
        self.batch_norm_before_output = batch_norm_before_output
        self._input_shape = input_shape
        self._output_shape = output_shape
        self.output_activation = output_activation
        self.name = name
        self.regularization_value = regularization_value

    def get_input_output_tensors(self):
        def add_layer(input_tensor, is_last_hidden_layer=False):
            r = self.regularization_value
            if r is not None:
                kwargs = dict(kernel_regularizer=l1_l2(r, r), bias_regularizer=l1_l2(r, r))
            else:
                kwargs = {}
            output_tensor = Dense(self.width, **kwargs)(input_tensor)
            if (not self.batch_normalization) or (is_last_hidden_layer and not self.batch_norm_before_output):
                return Activation(self.base_activation)(output_tensor)

            if not self.batch_norm_after_activation:
                output_tensor = BatchNormalization()(output_tensor)
                output_tensor = Activation(self.base_activation)(output_tensor)
            else:
                output_tensor = Activation(self.base_activation)(output_tensor)
                output_tensor = BatchNormalization()(output_tensor)

            return output_tensor

        input_shape = self._input_shape
        output_shape = self._output_shape

        input_layer = Input(input_shape)
        if len(input_shape) > 1:
            output_layer = Flatten()(input_layer)
        else:
            output_layer = input_layer

        for d in range(self.depth):
            if self.skip_connections and d:
                output_layer = Add()([output_layer, add_layer(output_layer, is_last_hidden_layer=d == (self.depth - 1))])
            else:
                output_layer = add_layer(output_layer, is_last_hidden_layer=d == (self.depth - 1))

        output_layer = Dense(np.prod(output_shape), activation=self.output_activation)(output_layer)
        if len(output_shape) > 1:
            output_layer = Reshape(output_shape)(output_layer)

        return input_layer, output_layer

    def get_default_config(self):
        return dict(
            depth=1, width=32, base_activation='relu', skip_connections=False, batch_normalization=False,
            batch_norm_after_activation=True, batch_norm_before_output=False, input_shape=(28, 28),
            output_shape=(10,), output_activation='softmax', name=None
        )

    def __str__(self):
        s = 'fully_connected-'
        for key in ['depth', 'width']:
            s += key + '_' + str(self.config[key]) + '-'
        return s[:-1]
