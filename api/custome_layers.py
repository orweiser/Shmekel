from keras.layers import Layer, GRUCell
# import keras.backend as K


class Gate(Layer):
    def __init__(self):
        super(Gate, self).__init__()

    def build(self, input_shape):
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, inputs, **kwargs):
        x, g = inputs

        return x * g

