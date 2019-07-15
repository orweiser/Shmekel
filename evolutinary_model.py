from json_maker import json_format
import json
import numpy as np
import os
from api import core
from layer_in_none_sequential import NoneSequentialLayer

LAYERS_TYPES = ['KerasLSTM', 'Dense']
ACTIVATION_FUNCTIONS = ['relu', 'sigmoid', 'tanh']


class ModelDict:
    def __init__(self, model=None):
        self.layers = []
        if model is None:
            self.model = 'non_sequential_architecture'
            self.num_of_layers = 0
            self.num_of_rnn_layers = 0
            self.output = NoneSequentialLayer(layer_type="dense", activation_function='softmax', connections=[],
                                              historical_number=float("inf"))
            self.input = NoneSequentialLayer(historical_number=0)
        else:
            self.model = model['model']
            self.num_of_layers = model['num_of_layers']
            self.num_of_rnn_layers = model['num_of_rnn_layers']
            for layer in model['layers']:
                self.layers.append(Layer(layer_dict=layer))
                connections = []
                for connection in self.layers[-1].connections:
                    for prev_layer in self.layers:
                        if prev_layer == connection:
                            connections.append(prev_layer)
                            break
                self.layers[-1].connections = connections

            self.output = NoneSequentialLayer(layer_type="dense", activation_function=model['output_activation'],
                                              connections=model['output_connections'], historical_number=float("inf"))
            self.input = NoneSequentialLayer(historical_number=0)

    def add_layer(self):
        self.num_of_layers += 1
        self.layers.append(NoneSequentialLayer())
        if self.layers[-1].type == 'KerasLSTM':
            self.num_of_rnn_layers += 1

        # connect the output of the new layer to a random layer
        np.random.choice(self.layers[:-1] + [self.output]).add_connection(self.layers[-1])

        # connect the input of the new layer to a random lower layer
        connection_added = False
        while connection_added is False:
            connection_added = self.layers[-1].add_connection(np.random.choice(self.layers[:-1] + [self.input]))

    def get_dict(self):
        layers = []
        self.layers.sort(key=lambda x: x.higher_layers, reverse=True)
        for layer in self.layers:
            layers.append(layer.get_dict())
        return {
            'model': self.model,
            'num_of_layers': self.num_of_layers,
            'num_of_rnn_layers': self.num_of_rnn_layers,
            'layers': layers,
            'output_layer': self.output.get_dict()
        }

    def crossover(self, other):
        this_layers = sorted(self.layers, key=lambda layer: layer.historical_number)
        that_layers = sorted(other.layers, key=lambda layer: layer.historical_number)
        new_layers = []
        this_index = 0
        that_index = 0
        while (this_index < len(this_layers)) & (that_index < len(that_layers)):
            new_layer = min(this_layers[this_layers].historical_number, that_layers[that_layers].historical_number)
            if this_layers[this_index] == new_layer:
                if that_layers[that_index] == new_layer:



class Generation:
    def __init__(self, gen_dict=None, generation=None, population_size=None):
        if gen_dict is not None:
            self.generation = gen_dict['generation']
            self.population_size = gen_dict['population_size']
        else:
            self.generation = generation
            self.population_size = population_size

    def get_dict(self):
        return {
            'generation': self.generation,
            'population_size': self.population_size
        }


def new_population(project_name, population_size):
    os.makedirs(r'.\Shmekel_Results\{project}\G0'.format(project=project_name), exist_ok=True)
    generation = Generation(generation=0, population_size=population_size)
    for i in range(population_size):
        model = ModelDict()
        model.add_layer()
        model_dict = model.get_dict()
        data = json_format('G0-{}'.format(i), model_config=model_dict)
        data['backup_config']['project'] = project_name
        if model_dict['num_of_rnn_layers'] == 0:
            if 'time_sample_length' in (data['train_dataset_config']):
                data['train_dataset_config']['time_sample_length'] = 1
            if 'time_sample_length' in (data['val_dataset_config']):
                data['val_dataset_config']['time_sample_length'] = 1
        file_name = r'.\Shmekel_Results\{project}\G0\config_{name}.json'.format(project=project_name, name=i)
        with open(file_name, 'w') as outfile:
            json.dump(data, outfile)
        file_name = r'.\Shmekel_Results\{project}\G0\G0.json'.format(project=project_name)
        with open(file_name, 'w') as outfile:
            json.dump(generation.get_dict(), outfile)


def train_generation(project_name, gen):
    generation = Generation(core.load_config(
            r'.\Shmekel_Results\{project}\G0\G{gen}.json'.format(project=project_name, gen=gen)))  # need to fill
    for net in range(generation.population_size):
        exp = core.get_exp_from_config(
                core.load_config(
                        r'.\Shmekel_Results\{project}\G{generation}\config_{net}.json'.format(project=project_name,
                                                                                              generation=gen,
                                                                                              net=net)))

        exp.run()

# main()
