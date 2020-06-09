import os
from copy import deepcopy
from api.core import get_exp_from_config, load_config
import pandas as pd
import numpy as np
import json
from api.core import Experiment

DENSE = 'Dense'
LSTM = 'KerasLSTM'


default_loss_config = {
    "loss": "categorical_crossentropy"
}
default_train_config = {
    "batch_size": 1024,
    "callbacks": None,
    "epochs": 200,
    "include_experiment_callbacks": True,
    "optimizer": "adam",
    "randomize": True,
    "steps_per_epoch": None,
    "validation_steps": None
}
default_backup_config = dict(handler='DefaultLocal')
default_train_dataset_config = dict(dataset='StocksDataset', val_mode=False, time_sample_length=7,
                                    stock_name_list=['fb'])
default_val_dataset_config = dict(dataset='StocksDataset', val_mode=True, time_sample_length=7,
                                  stock_name_list=['fb'])


DEFAULT = dict(
    backup_config=default_backup_config,
    loss_config=default_loss_config,
    # model_config=default_model_config,
    train_config=default_train_config,
    train_dataset_config=default_train_dataset_config,
    val_dataset_config=default_val_dataset_config
)


def _create_config(output_activation, layers_def, early_stop=True, dropout=True, dropout_rate=0.2):
    # neurons_combination[index]):

    depth = len(layers_def)
    model_config = {
        'model': 'General_RNN',
        'num_of_layers': depth,
        'layers': [],
        'num_of_rnn_layers': 0,
        "output_activation": output_activation,
        'dropout': dropout,
        'dropout_rate': dropout_rate
    }

    for index, (layer_type, size, activation) in enumerate(layers_def):
        layer_config = {'type': layer_type, 'size': size, 'activation_function': activation,
                        'name': '_'.join(['l%d' % index, layer_type, str(size), activation])}
        if layer_type == LSTM:
            model_config['num_of_rnn_layers'] += 1
        model_config['layers'].append(layer_config)

    # noinspection PyTypeChecker
    model_config['experiment_name'] = '--'.join(
        [layer_config['name'] for layer_config in model_config['layers']]
    )

    if early_stop:
        model_config['callbacks'] = 'early_stop'

    return model_config


def get_random_sample(layers_types, activation_functions, depths, neurons):
    from random import choice
    depth = choice(list(depths))
    return [map(choice, [layers_types, neurons, activation_functions]) for _ in range(depth)]


def generate_grid_model(layers_types=(DENSE, ), activation_functions=('relu', 'sigmoid', 'tanh'),
                        depth=(3, 5, 10, 15, 20), neurons=(8, 32, 128), output_activation='softmax',
                        dropout=True, dropout_rate=0.2, early_stop=True):
    combs = get_random_sample(layers_types, activation_functions, depth, neurons)
    return _create_config(output_activation, combs, early_stop=early_stop, dropout=dropout, dropout_rate=dropout_rate)


def grid_jason_maker(amount_of_experiments=-1, run_experiments=True, defaults_override=None, **kwargs):
    while amount_of_experiments == -1 or amount_of_experiments > 0:
        model_config = generate_grid_model(**kwargs)
        name = model_config.pop('experiment_name')
        name = give_nickname(name)

        data = deepcopy(DEFAULT)
        data.update(defaults_override or {})
        data.update(dict(
            model_config=model_config,
            name=os.path.join(VERSION, name),
        ))

        # if no rnn layers, data set should not be a time series
        if model_config['num_of_rnn_layers'] == 0:
            if 'time_sample_length' in data['train_dataset_config']:
                data['train_dataset_config']['time_sample_length'] = 1
            if 'time_sample_length' in data['val_dataset_config']:
                data['val_dataset_config']['time_sample_length'] = 1

        exp = get_exp_from_config(data)
        exp.backup()
        if run_experiments:
            exp.run()

        if amount_of_experiments > 0:
            amount_of_experiments -= 1


def give_nickname(name):
    name = name.replace('KerasLSTM', 'LS')
    name = name.replace('Dense', 'D')
    name = name.replace('relu', 'r')
    name = name.replace('tanh', 't')
    name = name.replace('sigmoid', 's')
    name = name.replace('_', '')
    name = name.replace('-', '')
    name = name.replace('l', 'L')

    return name


if __name__ == '__main__':

    # Set version name and parameters to create a new models group
    VERSION = 'version-0.0.3'
    config_path = os.path.join(os.pardir, 'Shmekel_Results', VERSION)

    depth_list = (3, 5)
    neurons_list = (8, 32)
    activation_functions = ('relu', 'tanh')
    dropout = False
    dropout_rate = 0.2
    version_parameters = {
        "EXP_CONFIG": None,
        "MAX_NUM_OF_LAYERS": max(depth_list),
        "LAYERS_TYPES": ['Dense'],
        "ALL_DENSE_SAME": False,
        "ACTIVATION_FUNCTIONS": activation_functions,
        'DROPOUT': dropout,
        "DROPOUT_RATE": dropout_rate,
        "NUM_OF_EPOCHS": default_train_config['epochs']
    }

    # Save version config in version folder
    if not os.path.exists(os.path.join(config_path, 'version_parameters')):
        with open(os.path.join(config_path, 'version_parameters'), 'w') as outfile:
            json.dump(version_parameters, outfile)

    # Run version
    grid_jason_maker(depth=depth_list, neurons=neurons_list, activation_functions=activation_functions,
                     dropout=dropout, dropout_rate=dropout_rate)

