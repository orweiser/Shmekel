from ..core import Model
from keras import Input
import keras.layers as layers
from ..custome_layers import Gate


class SequentialMultiPathDAG(Model):

    def init(self, *args, num_duplicates=None, **kwargs):
        self.num_duplicates = num_duplicates
        raise NotImplementedError

    @property
    def agent(self):
        raise NotImplementedError()

    @property
    def pre_agent(self):
        raise NotImplementedError()

    def sequential_layers_desc(self):
        raise NotImplementedError

    def get_input_output_tensors(self):
        def LAYER(layer_name, params):
            return getattr(layers, layer_name)(**params)

        layers = self.sequential_layers_desc()
        input = LAYER(*layers[0])
        output = LAYER(*layers[-1])
        layers = [[LAYER(*l) for _ in range(self.num_duplicates)] for l in layers[1:-1]]

        x = input
        y = self.pre_agent(input)
        h = None

        for group in layers:
            mask, h = self.agent(y, h)

            weighted = []
            for i, l in enumerate(group):
                weighted.append()

    def __str__(self):
        pass

    def get_default_config(self) -> dict:
        pass


class GatedSequential(Model):

    def init(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    def rnn_cell(self):
        raise NotImplementedError()

    @property
    def pre_rnn(self):
        raise NotImplementedError()

    def create_layers(self):
        raise NotImplementedError()

    def create_input_layers(self):
        raise NotImplementedError()

    def create_output_layer(self):
        # return layers.Dense(self._output_shape, activation='softmax')
        raise NotImplementedError()

    def get_input_output_tensors(self):
        input_layer, agent_input = self.create_input_layers()

        x = input_layer
        rnn_inputs = self.pre_rnn(agent_input)

        for layer in self.create_layers():
            x = layer(x)
            rnn_inputs = self.rnn_cell(rnn_inputs)
            g = rnn_inputs[0]

            x = Gate()([x, g])

        x = self.create_output_layer()(x)

        return x, [input_layer, agent_input]

    def __str__(self):
        pass

    def get_default_config(self) -> dict:
        raise NotImplementedError()


class GatedDense(GatedSequential):

    def init(self, *args, **kwargs):
        pass

    @property
    def rnn_cell(self):
        pass

    @property
    def pre_rnn(self):
        pass

    def create_layers(self):
        pass

    def create_input_layers(self):
        return input_layer, agent_input_layer

    def create_output_layer(self):
        return layers.Dense(self._output_shape[-1], activation='softmax')

    def get_default_config(self) -> dict:
        pass
