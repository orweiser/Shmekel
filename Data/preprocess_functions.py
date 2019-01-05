"""
just some basic preprocess and feature extraction functions.
we will have to create much more interesting features in the future, but it is enough to get a taste
"""
import numpy as np


pattern = ('Open', 'High', 'Low', 'Close')  # this is the pattern we use to describe the stocks in the numpy data set


def __return_bool_rise(rise, bool_type):
    """
    just an inner function, do not touch unless you want to add another way to represent the boolean variable
    of rise / no rise.
    :param rise: a boolean array with zeros for no rise and ones for rise
    :param bool_type: a string. one of the following
        'binary': returns the 1-D array: rise. use for binary classification
        'categorical': returns a 2-D array: [rise, 1 - rise]. use for categorical classification
    :return: a numpy array
    """
    if bool_type not in ['categorical', 'binary']:
        raise Exception('Unexpected value for parameter bool_type, got', bool_type)

    if bool_type == 'binary':
        return rise

    if bool_type == 'categorical':
        return np.stack([rise, 1 - rise], axis=-1)


def normalize_data_array(array, axis=0, regulation_value=1e-8):
    """
    a normalization function. returns an array with zero mean and variance 1.
    note: each feature is treated differently with its own mean and variance to compute
    """
    mean = np.mean(array, axis=axis)
    std = np.std(array, axis=axis)

    std[std < regulation_value] = regulation_value

    return (array - mean) / std


def feature_difference(array, keys=('Close', 'Open'), bool_type=None):
    """
    for each time stamp returns the difference determined by :param keys
    :param keys: a tuple with two keys. subtract keys[1] from keys[0]
    :param bool_type: if not None returns the boolean representation of the difference
    """
    diff = array[:, pattern.index(keys[0])] - array[:, pattern.index(keys[1])]

    if bool_type is None:
        return diff

    bool_rise = 1 * (diff > 0)
    return __return_bool_rise(bool_rise, bool_type)


def feature_ratio(array, keys=('Close', 'Open'), regulation_value=1e-8):
    """
    for each time stamp returns the ratio determined by :param keys
    :param keys: a tuple with two keys. divide keys[0] by keys[1]
    """
    numerator = array[:, pattern.index(keys[0])]
    denominator = array[:, pattern.index(keys[1])]

    denominator[denominator < regulation_value] = regulation_value
    return numerator / denominator


def concat_features(features_list):
    """
    gets a list of features and returns it as a single numpy array
    :param features_list: list of features (ordered)
    :return: a numpy array
    """
    new_list = []
    for feature in features_list:
        if len(feature.shape) > 2:
            raise Exception('Expected features to have only 1 or 2 dimensions, got shape', feature.shape)

        if len(feature.shape) == 2:
            new_list.append(feature)
        new_list.append(feature[:, np.newaxis])

    return np.concatenate(new_list, axis=-1)


def down_sample(array, factor=2, discard_oldest=True):
    """
    Down samples the data and keeping it on the same pattern and the meaning of 'Open', 'High', 'Low', 'Close'
    :param array: data array and NOT feature array. you should always down sample BEFORE feature extraction
    :param factor: down sampling factor
    :param discard_oldest: boolean. which samples to discard if num_samples is not a multiple of factor
        if True: discards oldest samples
        if False: discards latest samples
    :return: a down sampled array
    """
    num_samples = (array.shape[0] // factor) * factor

    if discard_oldest:
        array = array[-num_samples:]
    else:
        array = array[:num_samples]

    new_array = np.zeros((num_samples // factor,) + array.shape[1:])

    ind = {key: pattern.index(key) for key in pattern}
    for i, j in enumerate(range(0, num_samples, factor)):
        new_array[i, ind['Open']] = array[j, ind['Open']]
        new_array[i, ind['Close']] = array[j + (factor - 1), ind['Close']]

        new_array[i, ind['High']] = np.max(array[j:(j + factor), ind['High']])
        new_array[i, ind['Low']] = np.max(array[j:(j + factor), ind['Low']])

    return new_array


# todo: PCA, per stock and over all stocks

