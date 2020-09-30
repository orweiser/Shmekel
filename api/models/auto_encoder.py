from ..core.model import *
from keras.layers import Dense, Flatten, Activation, BatchNormalization, Add, Reshape
from keras import Input
import numpy as np
import math


class AutoEncoder(Model):
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

    def init(self, depth=1, width=32, base_activation='relu', skip_connections=False, batch_normalization=False,
             batch_norm_after_activation=True, batch_norm_before_output=False, input_shape=(28, 28),
             output_shape=(10,), output_activation='softmax', name=None):
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

    def get_input_output_tensors(self):
        def add_layer(layer_width, input_tensor, is_last_hidden_layer=False):
            output_tensor = Dense(layer_width)(input_tensor)
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
            # TODO - make this generic somehow
            actual_width = 4 if d == math.floor(self.depth / 2) else self.width
            if self.skip_connections and d:
                output_layer = Add()([output_layer, add_layer(actual_width, output_layer, is_last_hidden_layer=d == (self.depth - 1))])
            else:
                output_layer = add_layer(actual_width, output_layer, is_last_hidden_layer=d == (self.depth - 1))

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
        s = 'autoencoder-'
        for key in ['depth', 'width']:
            s += key + '_' + str(self.config[key]) + '-'
        return s[:-1]
