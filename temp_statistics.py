import pickle as pkl
import pandas as pd
import os
import json
import numpy as np
from api import core






def model_raw_stats(model_path):
    """

    :param model_path: model directory path
    :return: dictionary containing model timestamp, name, config, history and best epoch
    """

    model_name = 'config_' + model_path.split('/')[-2]
    model_timestamp = model_path.split('/')[-3]

    with open(model_path + 'config.json', 'r') as f:
        config = json.loads(f.read())

    h = os.listdir(model_path + 'histories/')
    with open(model_path + '/histories/{history}'.format(history=h[-1]), 'rb') as f:
        history = pkl.load(f)
    model_history = pd.DataFrame.from_dict(history)

    model = {"timestamp": model_timestamp,
             "name": config['name'],
             "model": config[''],
             "loss": loss_config,
             "history": model_history}
    return model


def get_models_raw_stats(base_dir):
    """

    :param base_dir: directory path containing the timestamps
    :return: list containing all models raw data in base_dir
    """
    models_raw_stats = []
    time_stamps_list = os.listdir(base_dir)
    for t in time_stamps_list:
        ts_dir = base_dir + '/{time_stamp}'.format(time_stamp=t)
        model_list = [d for d in os.listdir(ts_dir)
                      if os.path.isdir(os.path.join(ts_dir, d))]
        for m in model_list:
            hist_dir = base_dir + '/{time_stamp}/{model}/histories'.format(time_stamp=t, model=m)
            if os.path.isdir(hist_dir):
                model = model_raw_stats(ts_dir + '/{model}/'.format(model=m))
                models_raw_stats.append(model)

    return models_raw_stats


def get_model_stats(models_raw_stats, metric=None, config_keys=None):
    """

    :param config_keys: list of config to account for statistics
    :param models_raw_stats: list of ModelRawStats Instances
    :param metric: list of training metrics on which best epoch is determined
    :return: model statistics
    """
    metric = ['acc', 'loss', 'val_acc', 'val_loss'] or metric
    config_keys = ['num_of_layers', 'num_of_rnn_layers', 'dropout_rate'] or config_keys

    # TODO implement for key layers in model -> model config
    # convert raw data, excluding model history, to pandas DataFrame and add best epoch
    models_stats = pd.DataFrame(columns=['timestamp', 'name'] + config_keys + ['best_epoch'])
    for i, model in enumerate(models_raw_stats):
        s = pd.Series(index=['timestamp', 'name'] + config_keys + ['best_epoch'], name=i)
        s['timestamp'] = model['timestamp']
        s['name'] = model['name']
        for key in config_keys:
            s[key] = model['config'][key]

        s['best_epoch'] = get_best_epoch(model, metric)
        print(s)
        models_stats = models_stats.append(s)

    return models_stats


def get_best_epoch(model_raw_stats, metric_list):
    """

    :param models_raw_stats: list of ModelRawStats Instances
    :param metric_list: list of training metrics on which best epoch is determined
    :return: dictionary containing models best epochs stats
    """
    best_epochs = {}
    for metric in metric_list:
        if metric == 'acc' or metric == 'val_acc':
            max_ind = np.argmin(model_raw_stats['history'][metric])
            best_epochs.update({metric: [model_raw_stats['history'].iloc[max_ind], max_ind]})

        else:
            min_ind = np.argmin(model_raw_stats['history'][metric])
            best_epochs.update({metric: [model_raw_stats['history'].iloc[min_ind], min_ind]})

    return best_epochs


def get_convergence_epoch(data, metric_list):
    """

    :param models_raw_stats: list of ModelRawStats Instances
    :param metric_list: list of training metrics on which best epoch is determined
    :return: dictionary containing model convergences epochs, or False if convergence hasn't been achieved
    """
    convergence_epochs = {}
    for metric in metric_list:
        pass
    return convergence_epochs


def distribution(data, model_keys, metric_list):
    """

    :param data: models data
    :param model_keys: list of model config keys
    :param metric_list: list of training metrics
    :return: distribution according to model_keys
    """

    # keys_max_val = {}
    # for key in model_keys:
    #     max = max(raw_data['model'][:])


dif_dir = 'C:/Shmekel/local_repository/Shmekel_Results/default_project'
x = get_models_raw_stats(dif_dir)
y = get_model_stats(x)
z = 3
