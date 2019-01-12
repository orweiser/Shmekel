import numpy as np
import os
import pickle as pkl
from shmekel_config import _read_a_file, get_config, config_path


stock_types, pattern, feature_axis, data_path, stock_data_file_ending = [None] * 5

globals().update(**get_config())

if not data_path or not os.path.exists(data_path) or not os.listdir(data_path):
    msg = '\n"data_path" param from shmekel_config points to nowhere or to an empty folder. got:\n' + str(data_path) + \
          '\nPlease fill in details at: ' + config_path
    raise Exception(msg)


def tckt_to_file_path(tckt, stock_type='Stocks'):
    def assert_tckt(tckt):
        """
        :type tckt: str
        """
        tckt = tckt.split(os.path.sep)[-1]
        tckt = tckt.split(os.path.altsep)[-1]
        tckt = tckt.split('.')[0]
        return tckt

    return os.path.join(data_path, stock_type, assert_tckt(tckt) + '.us.txt')


def get_list_path(stock_type='Stocks'):
    return os.path.join(data_path, stock_type + stock_data_file_ending)


def __file_to_dict(fname):
    content = _read_a_file(fname)
    if not content:
        return False

    keys = content[0].split(',')
    log = {key: [] for key in keys}
    for line in content[1:]:
        for key, val in zip(keys, line.split(',')):
            if key == 'Date':
                log[key].append(tuple([int(v) for v in val.split('-')]))
            else:
                log[key].append(float(val))
    return log


def load_stock(tckt, stock_type='Stocks'):
    if stock_type not in stock_types:
        raise Exception('param stock_type not in stock_types. got ' + stock_type)

    log = __file_to_dict(tckt_to_file_path(tckt, stock_type))

    if not log:
        return log

    a = np.concatenate([log[key] for key in pattern], axis=feature_axis)

    return a, log['Date']


def _save_valid_stock_lists():
    for s_type in stock_types:
        tckts_list = []

        listdir = os.listdir(os.path.join(data_path, s_type))

        print('creating stock list -', s_type)
        for i, tckt in enumerate(listdir):
            tckt = tckt.split('.')[0]
            if not i % 100:
                print(i, '/', len(listdir), '   ', tckt)

            path = tckt_to_file_path(tckt, stock_type=s_type)
            log = __file_to_dict(path)
            if not log:
                continue

            num_samples = len(log['Date'])
            tckts_list.append((tckt, num_samples, path))

        list_path = get_list_path(s_type)
        with open(list_path, 'wb') as pickle_file:
            pkl.dump(tckts_list, pickle_file)

        print('Saved stock list -', s_type, 'at:\n' + list_path, end='\n\n')


def get_data_info(stock_type='Stocks'):
    """
    :param stock_type: one of ['ETFs', 'Stocks']
    :return: a list of tuples of the form (stock_name, num_time_stemps, numpy_file_path)
    """
    with open(get_list_path(stock_type), 'rb') as f:
        x = pkl.load(f)
    return x


def random_stock_generator(stock_type='Stocks', min_time_stamps_per_samples=1000, forever=False, randomize=False):
    """
    :param stock_type: 'ETFs' or 'Stocks'
    :param min_time_stamps_per_samples: don't return arrays with less then this parameter value number of lines
    :param forever: if True, going over all stocks forever, otherwise only once
    :param randomize: if True, shuffles the order of stocks
    :return: stocks numpy arrays, one by one
    """
    data_info = get_data_info(stock_type)
    if min_time_stamps_per_samples:
        data_info = [d for d in data_info if d[1] > min_time_stamps_per_samples]

    def ind_gen():
        while True:
            perm = range(len(data_info)) if not randomize else np.random.permutation(len(data_info))
            for i in perm:
                yield i

            if not forever:
                break

    for i in ind_gen():
        tckt, _, _ = data_info[i]
        yield load_stock(tckt, stock_type=stock_type)


if not all([os.path.exists(get_list_path(st)) for st in stock_types]):
    s = input('valid_stock_list is not saved, generate lists? [y/n]')
    if s == 'y':
        _save_valid_stock_lists()