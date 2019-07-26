import json
import numpy as np
from api import core
import time
import os
from itertools import chain, combinations, combinations_with_replacement, product
from copy import deepcopy as copy
from api.core import get_exp_from_config, load_config

from api.core.grid_search import GridSearch2

EXP_CONFIG = None  # todo
MIN_SIZE = 3
MAX_SIZE = 7
MAX_NUM_OF_LAYERS = 5
LAYERS_TYPES = ['KerasLSTM', 'Dense']
ALL_DENSE_SAME = False
ACTIVATION_FUNCTIONS = ['relu', 'sigmoid', 'tanh']
DROPOUT_CHANCE = 0.7
MIN_DROPOUT = 0.1
MAX_DROPOUT = 0.3
DENSE = 'Dense'
LSTM = 'KerasLSTM'
MAX_DEPTH = 20
EARLY_STOP = True

name2num = {'Yishai': 0, 'Michael': 1, 'Ron': 2, 'Rotem': 3}


def get_config_identifiers(model_config):
    identifiers = dict()

    identifiers['model'] = model_config['model']
    identifiers['num_of_layers'] = model_config['num_of_layers']
    identifiers['num_of_rnn_layers'] = model_config['num_of_rnn_layers']
    identifiers["output_activation"] = model_config['output_activation']

    # print(model_config['layers'])
    for i, layer in enumerate(model_config['layers']):
        identifiers['layer_%d_type' % i] = layer['type']
        identifiers['layer_%d_size' % i] = layer['size']
        if layer['type'] == DENSE:
            identifiers['layer_%d_activation' % i] = layer['activation_function']
        else:
            identifiers['layer_%d_activation' % i] = None

    for j in range(i + 1, 10):
        # print(model_config['layers'])
        identifiers['layer_%d_type' % j] = None
        identifiers['layer_%d_size' % j] = None
        identifiers['layer_%d_activation' % j] = None

    return identifiers


def generate_grid_models(layers_types=[LSTM, DENSE], activation_functions=['relu', 'sigmoid', 'tanh'], min_depth=3,
                         max_depth=5, neurons=[], output_activation='softmax'):
    models_configs = []
    for depth in range(min_depth, max_depth + 1):
        for layers_combination in list(product(layers_types, repeat=depth)):
            for neurons_combination in list(product(neurons, repeat=depth)):
                model_config = {
                    'model': 'General_RNN',
                    'num_of_layers': depth,
                    'layers': [],
                    'num_of_rnn_layers': 0,
                    "output_activation": output_activation
                }
                name = ''
                for index, layer in enumerate(layers_combination):
                    if layer == DENSE:
                        layer_config = None
                    else:
                        layer_config = {
                            'type': layer,
                            'size': neurons_combination[index],
                        }
                        layer_config['name'] = '_'.join([layer_config['type'], str(layer_config['size'])])
                        name += layer_config['name'] + '_'
                        model_config['num_of_rnn_layers'] += 1
                    model_config['layers'].append(layer_config)
                num_of_dense_layers = model_config['layers'].count(None)
                if EARLY_STOP:
                    model_config['callbacks'] = 'early_stop'
                if num_of_dense_layers is 0:
                    model_config['name'] = name
                    models_configs.append(model_config)
                else:
                    for activation_functions_combinations in list(
                            product(activation_functions, repeat=num_of_dense_layers)):
                        new_model_config = {
                            'model': 'General_RNN',
                            'num_of_layers': depth,
                            'layers': [],
                            'num_of_rnn_layers': model_config['num_of_rnn_layers'],
                            "output_activation": output_activation
                        }
                        index = 0
                        name = ''
                        for layer_index, layer in enumerate(model_config['layers']):
                            if layer is None:
                                layer_config = {
                                    'type': DENSE,
                                    'size': neurons_combination[layer_index],
                                    'activation_function': activation_functions_combinations[index]
                                }
                                index += 1
                                name += '_'.join([
                                    layer_config['type'], str(layer_config['size']), layer_config['activation_function']
                                ]) + '_'
                                new_model_config['layers'].append(layer_config)
                            else:
                                name += layer['name'] + '_'
                                new_model_config['layers'].append(layer)
                        if EARLY_STOP:
                            new_model_config['callbacks'] = 'early_stop'
                        new_model_config['name'] = name
                        models_configs.append(new_model_config)
    return models_configs


def generate_model_config(output_activation='softmax'):
    model_config = {
        'model': 'General_RNN',
        'num_of_layers': np.random.randint(1, MAX_NUM_OF_LAYERS),
        'num_of_rnn_layers': 0,
        'layers': [],
        "output_activation": output_activation
    }

    # Choose if adding dropout
    dropout = np.random.random()
    model_config['dropout'] = False if dropout > DROPOUT_CHANCE else True
    if model_config['dropout']:
        # choose dropout rate
        model_config['dropout_rate'] = np.random.uniform(MIN_DROPOUT, MAX_DROPOUT)

    # add activation function to Dense layers
    activation_function = np.random.choice(ACTIVATION_FUNCTIONS) if ALL_DENSE_SAME else None

    # create model layers
    for layer in range(model_config['num_of_layers']):
        model_config['layers'].append({
            'type': np.random.choice(LAYERS_TYPES),
            'size': int(2 ** np.random.random_integers(MIN_SIZE, MAX_SIZE))
        })
        if model_config['layers'][layer]['type'] == 'Dense':
            model_config['layers'][layer]['activation_function'] = activation_function or np.random.choice(
                    ACTIVATION_FUNCTIONS)
        else:
            model_config['num_of_rnn_layers'] += 1

    return model_config


def json_format(name, backup_config=None, loss_config=None, model_config=None, train_config=None,
                val_dataset_config=None, train_dataset_config=None):
    data_dic = {}
    data_dic['backup_config'] = backup_config or {
        "handler": "DefaultLocal",
        "history_backup_delta": 1,
        "project": "default_project",
        "save_history_after_training": True,
        "save_snapshot_after_training": True,
        "snapshot_backup_delta": 1
    }
    data_dic['loss_config'] = loss_config or {
        "loss": "categorical_crossentropy"
    }
    data_dic['model_config'] = model_config or {
        # "input_shape": [
        #     7,
        #     4
        # ],
        "model": "LSTM",
        "output_activation": "softmax",
        "output_shape": [
            2
        ],
        "units": 128
    }
    data_dic['name'] = name
    data_dic['train_config'] = train_config or {
        "augmentations": None,
        "batch_size": 1024,
        "callbacks": None,
        "epochs": 20,
        "include_experiment_callbacks": True,
        "optimizer": "adam",
        "randomize": True,
        "steps_per_epoch": None,
        "validation_steps": None
    }
    data_dic['train_dataset_config'] = train_dataset_config or {
        "config_path": None,
        "dataset": "StocksDataset",
        "feature_list": None,
        "output_feature_list": None,
        "stock_name_list": None,
        "time_sample_length": 16,
        "val_mode": True

    }
    data_dic['val_dataset_config'] = val_dataset_config or {
        "config_path": None,
        "dataset": "StocksDataset",
        "feature_list": None,
        "output_feature_list": None,
        "stock_name_list": None,
        "time_sample_length": 16,
        "val_mode": True
    }

    # json_data = json.dumps(data_dic)
    # return json_data
    return data_dic


def create_configs_directory():
    timestamp = round(time.time())
    main_dir = os.pardir
    main_dir = os.path.join(main_dir, 'Shmekel_Results')
    main_dir = os.path.join(main_dir, 'default_project')
    main_dir = os.path.join(main_dir, str(timestamp))
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
    return main_dir


def random_jason_maker(main_dir):
    for i in range(100):
        model_name = 'RNN_MODEL_{test_number}'.format(test_number=i)
        dir_name = os.path.join(main_dir, model_name)
        file_name = os.path.join(dir_name, 'config_{name}.json'.format(name=model_name))
        figpath_train = os.path.join(dir_name, 'train_fig_{name}.png'.format(name=model_name))
        figpath_val = os.path.join(dir_name, 'val_fig_{name}.png'.format(name=model_name))

        exp_model_name = os.path.join(str(timestamp), model_name)
        model = generate_model_config()
        data = json_format(model_config=model, name=exp_model_name,
                           train_dataset_config=dict(dataset='SmoothStocksDataset', val_mode=False,
                                                     figpath=figpath_train,
                                                     time_sample_length=7),
                           val_dataset_config=dict(dataset='SmoothStocksDataset', val_mode=True, figpath=figpath_val,
                                                   time_sample_length=7),
                           )
        if model['num_of_rnn_layers'] == 0:
            if 'time_sample_length' in data['train_dataset_config']:
                data['train_dataset_config']['time_sample_length'] = 1
            if 'time_sample_length' in data['val_dataset_config']:
                data['val_dataset_config']['time_sample_length'] = 1
        os.mkdir(dir_name)
        with open(file_name, 'w') as outfile:
            json.dump(data, outfile)


# exp1 = core.get_exp_from_config(core.load_config(file_name))
#     print(exp1.model.summary())
#     # exp1.run()


def grid_jason_maker(main_dir):
    models = generate_grid_models(neurons=[8, 32, 128], activation_functions=['relu'])
    print(len(models))
    for i, model_config in enumerate(models):
        model_name = model_config['name']
        dir_name = os.path.join(main_dir, model_name)
        file_name = os.path.join(main_dir, 'config_{name}.json'.format(name=model_name))
        figpath_train = os.path.join(dir_name, 'train_fig_{name}.png'.format(name=model_name))
        figpath_val = os.path.join(dir_name, 'val_fig_{name}.png'.format(name=model_name))

        name = model_config.pop('name')
        data = json_format(model_config=model_config, name=name,
                           train_dataset_config=dict(dataset='SmoothStocksDataset', val_mode=False,
                                                     figpath=figpath_train,
                                                     time_sample_length=7),
                           val_dataset_config=dict(dataset='SmoothStocksDataset', val_mode=True, figpath=figpath_val,
                                                   time_sample_length=7),
                           )

        data['identifiers'] = get_config_identifiers(model_config)

        if model_config['num_of_rnn_layers'] == 0:
            if 'time_sample_length' in data['train_dataset_config']:
                data['train_dataset_config']['time_sample_length'] = 1
            if 'time_sample_length' in data['val_dataset_config']:
                data['val_dataset_config']['time_sample_length'] = 1

        with open(file_name, 'w') as outfile:
            json.dump(data, outfile)
    return main_dir


def train_by_modulo(name):
    gs = GridSearch2(grid_jason_maker(create_configs_directory()))
    for exp in gs.iter_modulo(rem=name2num[name]):
        print(exp)
        exp.run()


def print_statistics(path, compare, fixed_values={}, metric='val_acc', file=None):
    results = {}
    for dir in os.listdir(path):
        if 'histories' in os.listdir(os.path.join(path, dir)):
            config = load_config(os.path.join(path, dir))
            exp = get_exp_from_config(config)
            config = load_config(os.path.join(path, '1563481881', 'config_' + dir + '.json'))
            identifiers = config.pop('identifiers')
            exp.identifiers = identifiers
            is_in_fixed_values = True
            for key, value in fixed_values.items():
                is_in_fixed_values = exp.identifiers[key] == value
            if is_in_fixed_values:
                if compare == "size":
                    size = 0
                    for layer in range(exp.identifiers['num_of_layers']):
                        size += exp.identifiers['layer_{layer}_size'.format(layer=layer)]
                    if size not in results:
                        results[size] = []
                    results[size].append(exp.results.get_best_epoch()[metric])
                else:
                    if exp.identifiers[compare] not in results:
                        results[exp.identifiers[compare]] = []
                    results[exp.identifiers[compare]].append(exp.results.get_best_epoch()[metric])

    for key, value in results.items():
        best = np.max(value)
        size = len(value)
        avg = np.mean(value)
        worst = np.min(value)
        print('Slice - {compare} {key}'.format(compare=compare, key=str(key)))
        print('total nets: %d;  max: %f;   avg: %f; min: %f' % (size, best, avg, worst))
        if file:
            os.system('echo Slice - {compare} {key} > {file}'.format(compare=compare, key=str(key),
                                                                     file=os.path.join(os.pardir, file)))
            print('echo total nets: %d;  max: %f;   avg: %f; min: %f > %s' % (
            size, best, avg, worst, os.path.join(os.pardir, file)))


# gs = GridSearch2('F:\\Users\\Ron\\Shmekels\\Shmekel_Results\\default_project\\1563206162')
# for exp in gs.iter_fixed({'num_of_layers': 1}):
#     exp.run()

# gs.plot_all_parameters_slices(figs_dir='C:\\Shmekel\\local_repository\\Shmekel_Results\\default_project\\Figs')
# for exp in gs.iter_modulo(rem=2):
#     exp.run()

# main
print_statistics('F:\\Users\\Ron\\Shmekels\\Shmekel_Results\\default_project', 'size',
                 file='results.txt')
