import os
import pandas as pd
import json

from api import core
from api.core.results import Results
from statistics import get_models_raw_stats, get_model_stats


def create_identifiers_csv(pth, parameters, metric=('val_acc',)):

    print("Creating Results CSV")

    # create DataFrame labels
    df = pd.DataFrame(columns=['name', 'num_of_layers', 'num_of_rnn_layers'])
    for i in range(parameters["MAX_NUM_OF_LAYERS"]):
        df['layer{num} type'.format(num=i + 1)] = ''
        df['layer{num} size'.format(num=i + 1)] = ''
        df['layer{num} activation_function'.format(num=i + 1)] = ''
    for m in metric:
        df['best epoch number by {metric}'.format(metric=m)] = ''
        df['best epoch values by {metric}'.format(metric=m)] = ''
    df['status'] = ''

    # Looping over all config files in the directory and appending the model config to the DataFrame
    exp_list = os.listdir(pth)
    for exp_name in exp_list:
        exp_config_full_path = os.path.join(pth, exp_name, 'config.json')
        histories_path = os.path.join(pth, exp_name, 'histories')

        if os.path.exists(exp_config_full_path) and os.path.exists(histories_path):
            with open(exp_config_full_path, 'r') as f:
                config = json.load(f)
            df = append_exp_to_df(df, config, os.path.join(pth, exp_name), parameters, metric)

    # Fill all empty cells with None
    df.fillna(value=-1, inplace=True)

    if not os.path.exists(os.path.join(pth, 'Results')):
        os.mkdir(os.path.join(pth, 'Results'))
    df.to_csv(os.path.join(pth, 'Results', 'grid_results'), index=False)

    print("DONE!")


def append_exp_to_df(df, config, exp_dir, parameters, metric):
    df_dict = {}
    for col in df:
        df_dict[col] = None

    df_dict['name'] = config['name']
    df_dict['num_of_layers'] = config['model_config']['num_of_layers']
    df_dict['num_of_rnn_layers'] = config['model_config']['num_of_rnn_layers']

    for i, layer in enumerate(config['model_config']['layers']):
        df_dict['layer{num} type'.format(num=i + 1)] = layer['type']
        df_dict['layer{num} size'.format(num=i + 1)] = layer['size']
        if 'activation_function' in layer:
            df_dict['layer{num} activation_function'.format(num=i + 1)] = layer['activation_function']

    exp = core.get_exp_from_config(config)
    for m in metric:
        df_dict['best epoch number by {metric}'.format(metric=m)] = exp.results.get_best_epoch_number(metric=m)
        df_dict['best epoch values by {metric}'.format(metric=m)] = exp.results.get_best_epoch(metric=m).scores[m]

    n_complited_epoch = len(os.listdir(os.path.join(exp_dir, 'histories')))
    if n_complited_epoch < parameters["NUM_OF_EPOCHS"]:
        df_dict['status'] = 'waiting'
    else:
        df_dict['status'] = 'Done'
    df = df.append(pd.Series(df_dict), ignore_index=True)

    return df


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


if __name__ == '__main__':

    VERSION = 'version-0.0.1'
    config_path = os.path.join(os.pardir, 'Shmekel_Results', VERSION)

    # Load version configuration
    with open(os.path.join(config_path, 'version_parameters')) as json_file:
        parameters = json.load(json_file)

    # Create version data set
    metric = ('acc', 'loss', 'val_acc', 'val_loss')
    grid_results_path = os.path.join(config_path, 'Results', 'grid_results')
    if not os.path.exists(grid_results_path):
        create_identifiers_csv(config_path, parameters=parameters, metric=metric)

    exp_results = pd.read_csv(grid_results_path)
    z=3


