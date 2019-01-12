"""
Hi everyone,
This file contains functions that will help you create the numpy data set and read the data easily.
Make sure this file is in the same folder wit the folders 'Stocks' and 'ETFs' or go over the code to fix it.

The numpy arrays have shape (num_time_stemps, 4)
    each time stamp is a vector of 4 values: ('Open', 'High', 'Low', 'Close', 'Volume')

***Do the following only once!!!***
    create the numpy data set by running the command 'create_numpy_dataset()'.

To read the data, use the generator 'data_reader()'. if you are not familiar with generators, here is a code example:
    gen = data_reader()
    stock1_numpy_array = next(gen)
    stock2_numpy_array = next(gen)
        ...
    stock100_numpy_array = next(gen)

To get a specific stock array by name use:
    stock_array = get_a_stock_array(stock_name)

To get list of available stocks, use 'get_data_info()'
    it returns a list of tuples of the form (name, num_time_stemps, numpy_file_path)

Have fun
"""
import numpy as np
import os
import pickle as pkl


def __read_a_file(fname):
    with open(fname) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    return [x.strip() for x in content]


def __file_to_dict(fname):
    content = __read_a_file(fname)
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


def __file_to_np_array(fname, reverse=False, return_dates=False, pattern=('Open', 'High', 'Low', 'Close', 'Volume')):
    f_dict = __file_to_dict(fname)
    if not f_dict:
        return None

    if reverse:
        for _, item in f_dict.items():
            item.reverse()

    array = np.zeros((len(f_dict[pattern[0]]), len(pattern)))

    for i, p in enumerate(pattern):
        array[:, i] = f_dict[p]

    if return_dates:
        return array, f_dict['Dates']
    return array


def create_numpy_dataset():
    for s_type in stock_types:
        tckts_list = []
        root = s_type + '_np/'
        if not os.path.exists(root):
            os.makedirs(root)

        listdir = os.listdir(s_type)
        for i, stock in enumerate(listdir):
            print(i, '/', len(listdir), '   ', stock)
            array = __file_to_np_array(s_type + '/' + stock, )
            if array is None:
                continue

            np_path = root + stock + '.npy'
            num_samples = array.shape[0]
            tckts_list.append((stock.split('.')[0], num_samples, np_path))

            np.save(np_path, array)

        with open(root[:-1] + '_list.pickle', 'wb') as pickle_file:
            pkl.dump(tckts_list, pickle_file)


def get_data_info(stock_type='Stocks'):
    """
    :param stock_type: one of ['ETFs', 'Stocks']
    :return: a list of tuples of the form (stock_name, num_time_stemps, numpy_file_path)
    """
    with open(stock_type + '_np_list.pickle', 'rb') as f:
        x = pkl.load(f)
    return x


def get_a_stock_array(stock_name, stock_type='Stocks'):
    """
    :param stock_type: 'ETFs' or 'Stocks'
    :return: that stock numpy array
    """
    np_path = stock_type + '_np/' + stock_name.split('.')[0] + '.us.txt.npy'
    return np.load(np_path)


def data_reader(stock_type='Stocks', min_time_stamps_per_samples=1000, forever=False, randomize=False):
    """
    :param stock_type: 'ETFs' or 'Stocks'
    :param min_time_stamps_per_samples: don't return arrays with less then this parameter value number of lines
    :param forever: if True, going over all stocks forever, otherwise only once
    :param randomize: if True, shuffles the order of stocks
    :return: stocks numpy arrays, one by one
    """
    data_info = get_data_info(stock_type)
    if min_time_stamps_per_samples:
        x = []
        for ind, bool_val in enumerate([d[1] > min_time_stamps_per_samples for d in data_info]):
            x.append(data_info[ind])
        data_info = x

    def ind_gen():
        while True:
            perm = range(len(data_info)) if not randomize else np.random.permutation(len(data_info))
            for i in perm:
                yield i

            if not forever:
                break

    for i in ind_gen():
        _, _, path = data_info[i]
        yield np.load(path)


stock_types = ['ETFs', 'Stocks']

create_numpy_dataset()