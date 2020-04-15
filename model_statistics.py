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

    if not os.path.exists(os.path.join(pth, 'Results')):
        os.mkdir(os.path.join(pth, 'Results'))
    df.to_csv(os.path.join(pth, 'Results', 'grid_results'), index=False)
    print("DONE!")


def append_exp_to_df(df, config, exp_dir, parameters, metric):

    df_labels = {}
    for col in df:
        df_labels[col] = None

    df_labels['name'] = config['name']
    df_labels['num_of_layers'] = config['model_config']['num_of_layers']
    df_labels['num_of_rnn_layers'] = config['model_config']['num_of_rnn_layers']

    for i, layer in enumerate(config['model_config']['layers']):
        df_labels['layer{num} type'.format(num=i + 1)] = layer['type']
        df_labels['layer{num} size'.format(num=i + 1)] = layer['size']
        if 'activation_function' in layer:
            df_labels['layer{num} activation_function'.format(num=i + 1)] = layer['activation_function']

    # exp = core.get_exp_from_config(core.load_config(exp_dir))
    # model_timestamp = '123123'  # FAKE TIMESTAMP
    # model_raw_stats = {
    #          "timestamp": model_timestamp,
    #          "name": exp.name,
    #          "model": exp.model_config,
    #          "results": exp.results,
    #          "loss": exp.loss_config}
    # model_stats = get_model_stats(model_raw_stats, metrics=metric)
    # for m in metric:
    #     df['best epoch number by {metric}'.format(metric=m)] = ''
    #     df['best epoch values by {metric}'.format(metric=m)] = ''

    n_complited_epoch = len(os.listdir(os.path.join(exp_dir, 'histories')))
    if n_complited_epoch < parameters["NUM_OF_EPOCHS"]:
        df_labels['status'] = 'waiting'
    else:
        df_labels['status'] = 'Done'
    df = df.append(pd.Series(df_labels), ignore_index=True)

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
    metric = ('val_acc',)
    grid_results_path = os.path.join(config_path, 'Results', 'grid_results')
    if not os.path.exists(grid_results_path):
        create_identifiers_csv(config_path, parameters=parameters)

    # Insert missing experiment data to data set
    exp_results = pd.read_csv(grid_results_path)
    # experiment_list = os.listdir(config_path)
    # for experiment_name in experiment_list:
    #     config_full_path = os.path.join(config_path, experiment_name, 'config.json')
    #     history_path = os.path.join(config_path, experiment_name, 'histories')
    #
    #     if os.path.exists(config_full_path) and os.path.exists(history_path):
    #         pass


