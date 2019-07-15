import numpy as np

LAYERS_TYPES = ['KerasLSTM', 'Dense']
ACTIVATION_FUNCTIONS = ['relu', 'sigmoid', 'tanh']


class NoneSequentialLayer:
    last_historical_number = 0

    def __init__(self, layer_type=None, size=1, historical_number=None, higher_layers=None, connections=None,
                 activation_function=None, layer_dict=None):
        if layer_dict:
            layer_type = layer_dict['type']
            activation_function = layer_dict['activation_function']
            size = layer_dict['size']
            historical_number = layer_dict['historical_number']
            higher_layers = layer_dict['higher_layers']
            connections = layer_dict['connections']
        self.type = layer_type or np.random.choice(LAYERS_TYPES)
        self.activation_function = activation_function
        if self.activation_function is None and self.type == 'Dense':
            self.activation_function = np.random.choice(ACTIVATION_FUNCTIONS)
        self.size = size
        if historical_number is None:
            NoneSequentialLayer.last_historical_number += 1
            historical_number = NoneSequentialLayer.last_historical_number
        self.historical_number = historical_number
        self.higher_layers = higher_layers or []
        self.connections = connections or []

    def __repr__(self):
        return str(self.historical_number)

    def add_connection(self, other):
        if self.historical_number == float('inf') or other.historical_number == 0:
            # connecting layer to output or input layer is always legal
            self.connections.append(other)
            return True

        if self == other:
            # layer cannot be connected to itself
            return False

        for layer in self.higher_layers:
            # making sure new connection isn't making a layer loop
            if other == layer:
                return False
        self.connections.append(other)
        other.update_higher_layers(self)

    def update_higher_layers(self, other):
        if self.historical_number == 0:
            # input layer doesn't need to update
            return

        for layer in self.higher_layers:
            # if the layer is already registered, there is no need to register it again
            if other == layer:
                return
        self.higher_layers.append(other.historical_number)

        # for other_layer in other.higher_layers:
        #     in_higher_layers = False
        #     for layer in self.higher_layers:
        #         if layer == other_layer:
        #             in_higher_layers = True
        #             break
        #     if in_higher_layers is False:
        #         self.higher_layers.append(other_layer)

        for layer in other.higher_layers:
            if layer not in self.higher_layers:
                self.higher_layers.append(layer.historical_number)

        for layer in self.connections:
            layer.update_higher_layers(other)

    def get_dict(self):
        higher_layers = []
        for layer in self.higher_layers:
            higher_layers.append(layer)
        connections = []
        for layer in self.connections:
            connections.append(layer.historical_number)
        return {
            'type': self.type,
            'activation_function': self.activation_function,
            'size': self.size,
            'historical_number': self.historical_number,
            'higher_layers': higher_layers,
            'connections': connections
        }

    def __eq__(self, other):
        if isinstance(other, NoneSequentialLayer):
            return self.historical_number == other.historical_number
        return self.historical_number == other
