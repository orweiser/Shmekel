import os
from copy import deepcopy
from api.core import get_exp_from_config


DENSE = 'Dense'
LSTM = 'KerasLSTM'
VERSION = 'version-0.0.1'


default_loss_config = {
    "loss": "categorical_crossentropy"
}
default_train_config = {
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
default_backup_config = dict(handler='DefaultLocal')
default_train_dataset_config = dict(dataset='StocksDataset', val_mode=False, time_sample_length=7)
default_val_dataset_config = dict(dataset='StocksDataset', val_mode=True, time_sample_length=7)


DEFAULT = dict(
    backup_config=default_backup_config,
    loss_config=default_loss_config,
    # model_config=default_model_config,
    train_config=default_train_config,
    train_dataset_config=default_train_dataset_config,
    val_dataset_config=default_val_dataset_config
)


def _create_config(output_activation, layers_def, early_stop=True):
    # neurons_combination[index]):

    depth = len(layers_def)
    model_config = {
        'model': 'General_RNN',
        'num_of_layers': depth,
        'layers': [],
        'num_of_rnn_layers': 0,
        "output_activation": output_activation
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


def get_random_sample(layers_types, activation_functions, min_depth, max_depth, neurons):
    from random import choice
    depth = choice(list(range(min_depth, max_depth + 1)))
    return [map(choice, [layers_types, neurons, activation_functions]) for _ in range(depth)]


def generate_grid_model(layers_types=(LSTM, DENSE), activation_functions=('relu', 'sigmoid', 'tanh'), min_depth=3,
                         max_depth=5, neurons=(8, 32, 128), output_activation='softmax', early_stop=True):
    combs = get_random_sample(layers_types, activation_functions, min_depth, max_depth, neurons)
    return _create_config(output_activation, combs, early_stop=early_stop)


def grid_jason_maker(amount_of_experiments=-1, run_experiments=True, defaults_override=None, **kwargs):
    while amount_of_experiments == -1 or amount_of_experiments > 0:
        model_config = generate_grid_model(**kwargs)
        name = model_config.pop('experiment_name')

        data = deepcopy(DEFAULT)
        data.update(defaults_override or {})
        data.update(dict(
            model_config=model_config,
            name=os.path.join(VERSION, name),
        ))

        # if model_config['num_of_rnn_layers'] == 0:
        #     if 'time_sample_length' in data['train_dataset_config']:
        #         data['train_dataset_config']['time_sample_length'] = 1
        #     if 'time_sample_length' in data['val_dataset_config']:
        #         data['val_dataset_config']['time_sample_length'] = 1

        exp = get_exp_from_config(data)
        exp.backup()
        if run_experiments:
            exp.run()

        if amount_of_experiments > 0:
            amount_of_experiments -= 1


""" Rotem? """
# EXP_CONFIG = None
# MIN_SIZE = 3
# MAX_SIZE = 7
# MAX_NUM_OF_LAYERS = 5
# LAYERS_TYPES = ['KerasLSTM', 'Dense']
# ALL_DENSE_SAME = False
# ACTIVATION_FUNCTIONS = ['relu', 'sigmoid', 'tanh']
# DROPOUT_CHANCE = 0.7
# MIN_DROPOUT = 0.1
# MAX_DROPOUT = 0.3
# MAX_DEPTH = 20

# def print_statistics(path, compare, fixed_values={}, metric='val_acc', file=None):
#     results = {}
#     for dir in os.listdir(path):
#         if 'histories' in os.listdir(os.path.join(path, dir)):
#             config = load_config(os.path.join(path, dir))
#             exp = get_exp_from_config(config)
#             config = load_config(os.path.join(path, '1563481881', 'config_' + dir + '.json'))
#             identifiers = config.pop('identifiers')
#             exp.identifiers = identifiers
#             is_in_fixed_values = True
#             for key, value in fixed_values.items():
#                 is_in_fixed_values = exp.identifiers[key] == value
#             if is_in_fixed_values:
#                 if compare == "size":
#                     size = 0
#                     for layer in range(exp.identifiers['num_of_layers']):
#                         size += exp.identifiers['layer_{layer}_size'.format(layer=layer)]
#                     if size not in results:
#                         results[size] = []
#                     results[size].append(exp.results.get_best_epoch()[metric])
#                 else:
#                     if exp.identifiers[compare] not in results:
#                         results[exp.identifiers[compare]] = []
#                     results[exp.identifiers[compare]].append(exp.results.get_best_epoch()[metric])
#
#     for key, value in results.items():
#         best = np.max(value)
#         size = len(value)
#         avg = np.mean(value)
#         worst = np.min(value)
#         print('Slice - {compare} {key}'.format(compare=compare, key=str(key)))
#         print('total nets: %d;  max: %f;   avg: %f; min: %f' % (size, best, avg, worst))
#         if file:
#             os.system('echo Slice - {compare} {key} > {file}'.format(compare=compare, key=str(key),
#                                                                      file=os.path.join(os.pardir, file)))
#             print('echo total nets: %d;  max: %f;   avg: %f; min: %f > %s' % (
#                 size, best, avg, worst, os.path.join(os.pardir, file)))


# def create_identifiers_csv(pth, metric=('val_acc',)):
#     # create DataFrame labels
#     df = pd.DataFrame(columns=['name', 'num_of_layers', 'num_of_rnn_layers'])
#     for i in range(MAX_NUM_OF_LAYERS):
#         df['layer{num} type'.format(num=i + 1)] = ''
#         df['layer{num} size'.format(num=i + 1)] = ''
#         df['layer{num} activation_function'.format(num=i + 1)] = ''
#     for m in metric:
#         df['best epoch number by {metric}'.format(metric=m)] = ''
#         df['best epoch values by {metric}'.format(metric=m)] = ''
#     df['status'] = ''
#     # Creating a dictionary containing keys as in the DataFrame
#     milon = {}
#     for col in df:
#         milon[col] = None
#
#     # Looping over all config files in the directory and appending the model config to the DataFrame
#     config_list = os.listdir(pth)
#     config_list.sort()
#     for config_name in config_list:
#         if config_name.endswith('.json'):
#             with open(os.path.join(pth, config_name), 'r') as f:
#                 config = json.load(f)
#             milon['name'] = config['name']
#             milon['num_of_layers'] = config['model_config']['num_of_layers']
#             milon['num_of_rnn_layers'] = config['model_config']['num_of_rnn_layers']
#             for i, layer in enumerate(config['model_config']['layers']):
#                 milon['layer{num} type'.format(num=i + 1)] = layer['type']
#                 milon['layer{num} size'.format(num=i + 1)] = layer['size']
#                 if 'activation_function' in layer:
#                     milon['layer{num} activation_function'.format(num=i + 1)] = layer['activation_function']
#             milon['status'] = 'waiting'
#             df = df.append(pd.Series(milon), ignore_index=True)
#             milon = dict.fromkeys(milon, None)
#
#     df.to_csv(os.path.join(pth, 'grid_results'), index=False)
#     print("DONE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#
#
# def insert_values_to_csv_cells(pth, find_by, row_tags, data):
#     '''
#
#     :param pth: csv path
#     :param find_by: a string of one of the csv column label
#     :param row_tags: list containing some value to unikly specify certain rows
#     :param data: dictionary containing keys as the csv file columns name
#     :return: modify a csv file
#     '''
#
#     # Loading a csv file to a dataframe and modify the wanted values
#     csv_data = pd.read_csv(pth)
#     for r in row_tags:
#         idx = csv_data.loc[csv_data[find_by] == r].index.values.astype(int)[0]
#         for key, val in data.items():
#             csv_data.at[idx, key] = val
#         csv_data.at[idx, 'status'] = 'Done'
#
#     # save the modified data frame to a new csv file, delete the old file and rename the new file
#     csv_data.to_csv(pth + '_temp', index=False)
#     os.remove(pth)
#     os.rename(pth + '_temp', pth)


""""""


# ***************IMPORTANT CODE STARTS HERE***************************

# config_path = os.path.join(os.pardir, 'Shmekel_Results', 'default_project', 'configs')
# grid_jason_maker(config_path)  # -- use this to create the configs

# metric = ('val_acc',)
# grid_results_path = os.path.join(config_path, 'grid_results')
# if not os.path.exists(grid_results_path):
#     create_identifiers_csv(config_path)
#
# exp_results = pd.read_csv(grid_results_path)
# # for exp in gs.iter_modulo(rem=2):
# for exp_name in exp_results['name']:
#     config = load_config(os.path.join(config_path, 'config_' + exp_name + '.json'))
#     config.pop('identifiers')
#     exp = get_exp_from_config(config)
#     exp.run()
#     for m in metric:
#         num = exp.results.get_best_epoch_number()
#         val = exp.results.get_best_epoch()[m]
#         labels_value_dict = {'best epoch numebr by {metric}'.format(metric=m): num,
#                              'best epoch values by {metric}'.format(metric=m): val}
#         idx = exp_results.loc[exp_results['name'] == exp._name].index.values.astype(int)[0]
#         if exp_results.at[idx, 'status'] != 'Done':
#             insert_values_to_csv_cells(grid_results_path, find_by='name', row_tags=[exp._name], data=labels_value_dict)
#
# # main
# # print_statistics('C:\\Shmekel\\local_repository\\Shmekel_Results\\default_project', 'size',
# #                  file='results.txt')
