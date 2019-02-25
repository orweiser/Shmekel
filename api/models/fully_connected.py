from ..core.model import *
from keras.layers import Dense, Flatten, Activation, BatchNormalization, Add, Reshape
from keras import Input
import numpy as np


# def save_args(obj, defaults, kwargs):
def save_args(obj, defaults, kwargs):
    for k, v in defaults.items():
        if k in kwargs:
            v = kwargs[k]
        setattr(obj, k, v)


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
    dataset_name: str

    def init(self, **kwargs):
            save_args(self, self.get_default_config(), kwargs)

    def get_inputs_outputs(self):
        def add_layer(input_tensor, is_last_hidden_layer=False):
            output_tensor = Dense(self.width)(input_tensor)
            if (not self.batch_normalization) or (is_last_hidden_layer and not self.batch_norm_before_output):
                return Activation(self.base_activation)(output_tensor)

            if not self.batch_norm_after_activation:
                output_tensor = BatchNormalization()(output_tensor)
                output_tensor = Activation(self.base_activation)(output_tensor)
            else:
                output_tensor = Activation(self.base_activation)(output_tensor)
                output_tensor = BatchNormalization()(output_tensor)

            return output_tensor

        input_shape = self.input_shape or {'mnist': (28, 28)}[self.dataset_name]
        output_shape = self.output_shape or {'mnist': (10,)}[self.dataset_name]

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

        return dict(inputs=input_layer, outputs=output_layer)

    def get_default_config(self):
        return {'dataset_name': 'mnist', 'depth': 1, 'width': 32, 'base_activation': 'relu', 'skip_connections': False,
                'batch_normalization': False, 'batch_norm_after_activation': True, 'batch_norm_before_output': False,
                'input_shape': None, 'output_shape': None, 'output_activation': 'softmax', 'name': None}

    def __str__(self):
        return 'Fully Connected' + self.dataset_name
