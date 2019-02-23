from typing import Dict, Any

from ..core.model import *
from keras.layers import Dense, Flatten, Activation, BatchNormalization, Add, Reshape
from keras import Input
import numpy as np


# def save_args(obj, defaults, kwargs):
def save_args(obj, defaults, kwargs):
    for k, v in defaults.items():
        if k in kwargs: v = kwargs[k]
        setattr(obj, k, v)


class FullyConnected(Model):

    # define types to prevent init function's warnings
    output_activation: str
    batch_norm_before_output: bool
    batch_norm_after_activation: bool
    batch_normalization: bool
    skip_connections: bool
    base_activation: str
    width: int
    depth: int
    name: str
    dataset_name: str

    def init(self, **kwargs):
            # self.depth = depth
            # self.width = width
            # self.base_activation = base_activation
            # self.skip_connections = skip_connections
            # self.batch_normalization = batch_normalization
            # self.batch_norm_after_activation = batch_norm_after_activation
            # self.batch_norm_before_output = batch_norm_before_output
            # self.input_shape = input_shape
            # self.output_shape = output_shape
            # self.output_activation = output_activation
            # self.name = name

            params = defaults
            params.update(kwargs)

            defaults: Dict[Any, Any] = {}
            save_args(self, defaults, kwargs)

    def get_inputs_outputs(self):
        def add_layer(x, is_last_hidden_layer=False):
            x = Dense(self.width)(x)
            if (not self.batch_normalization) or (is_last_hidden_layer and not self.batch_norm_before_output):
                return Activation(self.base_activation)(x)

            if not self.batch_norm_after_activation:
                x = BatchNormalization()(x)
                x = Activation(self.base_activation)(x)
            else:
                x = Activation(self.base_activation)(x)
                x = BatchNormalization()(x)

            return x

        input_shape = self.input_shape or {'mnist': (28, 28)}[self.dataset_name]
        output_shape = self.output_shape or {'mnist': (10,)}[self.dataset_name]

        input_layer = Input(input_shape)
        if len(input_shape) > 1:
            x = Flatten()(input_layer)
        else:
            x = input_layer

        for d in range(self.depth):
            if self.skip_connections and d:
                x = Add()([x, add_layer(x, is_last_hidden_layer=d == (self.depth - 1))])
            else:
                x = add_layer(x, is_last_hidden_layer=d == (self.depth - 1))

        x = Dense(np.prod(output_shape), activation=self.output_activation)(x)
        if len(output_shape) > 1:
            x = Reshape(output_shape)(x)

        return dict(inputs=input_layer, outputs=x)

    def __str__(self):
        return 'Fully Connected' + self.dataset_name
