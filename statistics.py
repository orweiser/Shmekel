import pickle as pkl
import pandas as pd
import os
import json
import numpy as np
from api import core
from api.core.results import Results
import matplotlib.pyplot as plt



def model_raw_stats(model_path):
    """

    :param model_path: model directory path
    :return: dictionary containing model timestamp, name, config and history
    """
    model_timestamp = model_path.split('/')[-3]

    exp = core.get_exp_from_config(core.load_config(model_path))

    model = {"timestamp": model_timestamp,
             "name": exp.name,
             "model": exp.model_config,
             "results": exp.results,
             "loss": exp.loss_config}
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


def get_model_stats(models_raw_stats, metrics=None, metric_trends=None, config_keys=None):
    """

    :param config_keys: list of config to account for statistics
    :param models_raw_stats: list of ModelRawStats Instances
    :param metric: list of training metrics on which best epoch is determined
    :return: model statistics
    """
    metrics = ['acc', 'loss', 'val_acc', 'val_loss'] or metrics
    if ~metric_trends:
        if metrics == ['acc', 'loss', 'val_acc', 'val_loss']:
            metric_trends = ['raise', 'fall', 'raise', 'fall']
        else:
            metric_trends['raise'] * len(metrics)


    config_keys = ['num_of_layers', 'num_of_rnn_layers', 'num_of_dense_layers', 'dropout_rate'] or config_keys

    # TODO implement for key layers in model -> model config
    # convert raw data, excluding model history, to pandas DataFrame and add best epoch
    models_stats = pd.DataFrame(columns=['timestamp', 'name'] + config_keys)
    for i, model in enumerate(models_raw_stats):
        s = pd.Series(index=['timestamp', 'name'] + config_keys, name=i)
        s['timestamp'] = model['timestamp']
        s['name'] = model['name']
        for key in config_keys:
            if key is "num_of_dense_layers":
                s[key] = model['model']['num_of_layers'] - model['model']['num_of_rnn_layers']
            elif key in model['model']:
                s[key] = model['model'][key]

        for metric, trend in zip(metrics, metric_trends):
            best_epoch = model['results'].get_best_epoch(metric, trend)
            s["best {metric} num".format(metric=metric)] = best_epoch.epoch_num
            for key, val in best_epoch.scores.items():
                s["best {metric} {key}".format(metric=metric, key=key)] = val
            s["first_close_to_best_epoch_num {metric}".format(metric=metric)] =\
                get_first_close_to_best_epoch(model, metric, trend=trend).epoch_num
            s["first_close_to_best_epoch {metric}".format(metric=metric)] =\
                get_first_close_to_best_epoch(model, metric, trend=trend).scores[metric]

        models_stats = models_stats.append(s)

    return models_stats


def get_first_close_to_best_epoch(model, metric, trend):
    results = model['results']
    best_epoch = results.get_best_epoch(metric, trend)
    for epoch in results._epoch_list:
        if abs(epoch.scores[metric] - best_epoch.scores[metric]) < 0.03:
            return epoch


dif_dir = 'C:/Shmekel/local_repository/Shmekel_Results/default_project'
x = get_models_raw_stats(dif_dir)
data = get_model_stats(x)
data = data.sort_values(by=['num_of_rnn_layers'])
file_name = 'C:/Shmekel/local_repository/Shmekel_Results/model_group_results.csv'
data.to_csv(file_name)

# plt.scatter(data['num_of_rnn_layers'], data['best val_acc val_acc'])
# plt.show()
