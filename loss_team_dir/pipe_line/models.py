from keras.layers import Dense, Flatten, Activation, BatchNormalization, Add, Reshape
from keras import Model, Input
import numpy as np


def fully_connected(dataset_name='mnist', depth=1, width=32, base_activation='relu', skip_connections=False,
                    batch_normalization=False, batch_norm_after_activation=True, batch_norm_before_output=False,
                    input_shape=None, output_shape=None, output_activation='softmax', name=None):
    def add_layer(x, is_last_hidden_layer=False):
        x = Dense(width)(x)
        if (not batch_normalization) or (is_last_hidden_layer and not batch_norm_before_output):
            return Activation(base_activation)(x)

        if not batch_norm_after_activation:
            x = BatchNormalization()(x)
            x = Activation(base_activation)(x)
        else:
            x = Activation(base_activation)(x)
            x = BatchNormalization()(x)

        return x

    input_shape = input_shape or {'mnist': (28, 28)}[dataset_name]
    output_shape = output_shape or {'mnist': (10,)}[dataset_name]

    input_layer = Input(input_shape)
    if len(input_shape) > 1:
        x = Flatten()(input_layer)
    else:
        x = input_layer

    for d in range(depth):
        if skip_connections and d:
            x = Add()([x, add_layer(x, is_last_hidden_layer=d == (depth - 1))])
        else:
            x = add_layer(x, is_last_hidden_layer=d == (depth - 1))

    x = Dense(np.prod(output_shape), activation=output_activation)(x)
    if len(output_shape) > 1:
        x = Reshape(output_shape)(x)

    return Model(input_layer, x, name=name)



