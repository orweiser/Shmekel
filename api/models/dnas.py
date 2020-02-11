from ..core import Model
from keras import Input
import keras.layers as layers
from keras.backend import tile


class Tile(layers.Layer):
    def __init__(self, tiles, **kwargs):
        super(Tile, self).__init__(**kwargs)

        self.tiles = tiles

    def compute_output_shape(self, input_shape):
        return [i * j for i, j in zip(input_shape, self.tiles)]

    def call(self, inputs, **kwargs):
        return tile(inputs, self.tiles)


class SequentialSuperDenseModel(Model):

    graph_config = [
        {'num_groups': None, 'gate_params': None, 'concat_params': None, 'layer_params': None},
    ]
    _input_shape = None

    def init_agent(self):
        raise NotImplementedError

    def agent_step(self):
        raise NotImplementedError

    @staticmethod
    def reshape_mask(mask, level):
        num_groups = level['num_groups']
        units = level['layer_params']['units']

        assert not units % num_groups, '"units" must be a multiple of "num_groups".' \
                                       '\nGot %d and %d' % (units, num_groups)
        tiles = (1, units / num_groups)
        return Tile(tiles)(mask)

    @staticmethod
    def gate(x, mask, level):
        return layers.Multiply()([x, mask])

    def concat(self, x, level):
        raise NotImplementedError()

    def get_input_output_tensors(self):
        agent_input = self.init_agent()

        input_layer = Input(self._input_shape)
        x = input_layer
        for level in self.graph_config:
            layer = layers.Dense(**level['layer_params'])
            x = layer(x)

            mask = self.agent_step()
            mask = self.reshape_mask(mask=mask, level=level)

            x = self.gate(x, mask, level)

            x = self.concat(x, level)

            x = layers.Activation('relu')(x)

        return [input_layer, agent_input], x

    def init(self, *args, **kwargs):
        pass

    def __str__(self):
        return 'SequentialSuperDenseModel'

    def get_default_config(self) -> dict:
        pass








