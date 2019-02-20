from Utils.Data import *
from os import listdir
from Indicators import *


def model_fitting_data(num_of_samples, time_batch, s_list, f_list=None, labels_type=None):
    """
    Preparing the data to fit our model by stacking different stocks,
    ordering the data by time batches and creating the labels
    :param num_of_samples: total number of time samples
    :param time_batch: batch time length
    :param s_list: list of wanted stocks
    :param f_list: list of wanted features, if none' return all aviable features
    :param labels_type: type of labels format
    :return: features list and their labels
    """

    # Load stocks from s_list
    data_path = '../Shmekel_config.txt'
    stocks_info = DataReader(data_path).get_data_info()
    stocks_info = [s for s in stocks_info if s[0] in s_list]
    loaded_stocks = [i[0] for i in stocks_info]
    for i in s_list:
        if i not in loaded_stocks:
            raise ValueError(i + 'is not available in database')


    # Load features from f_list
    if f_list is None:
        indicators_path = '../FeatureSpace'
        f_list = [f[:-3] for f in listdir(indicators_path) if f.endswith('.py')]
        f_list.remove('__init__')
    # ... turn feature list from strings to classes

    # Create Stock subclasses
    stocks = []
    for i, s in enumerate(stocks_info):
        if s[1] < num_of_samples:
            raise ValueError( s[1] + 'stock number of samples is smaller than num_of_samples' )
        stocks.append(Stock(stock_tckt=s[0], feature_list=f_list))
        stocks[i].slice(t_start=stocks[i].temporal_size-num_of_samples, num_time_samples=num_of_samples)

    final_data = create_features(time_batch, stocks)

    up_threshold = 1.01
    down_threshold = 0.99
    if labels_type == 'tanh':
        final_labels = create_tanh_labels(stocks, num_of_samples, time_batch, up_threshold, down_threshold)
    elif labels_type == 'crossentropy':
        final_labels = create_crossentropy_labels(stocks, num_of_samples, time_batch, up_threshold, down_threshold)
    else:
        raise ValueError(labels_type + 'unrecognized labels type')

    return final_data, final_labels


def create_features(time_batch, stocks):
    # Create stocks features

    final_data = []
    num_of_samples = stocks[0].temporal_size
    for stock in stocks:
        features_data = np.vstack(stock.numerical_feature_list).transpose()
        for i in range(num_of_samples-time_batch):
            final_data.append(features_data[i:i + time_batch, :])

    return final_data


def create_tanh_labels(stocks, time_batch, up_threshold, down_threshold):

    labels = []
    for stock in enumerate(stocks):
        cnd = stock.data[0]
        yesterday = cnd[0][3]
        for i, today in enumerate(cnd):
            if today[3] > yesterday * up_threshold:
                labels.append(1)
            elif today[3] < yesterday * down_threshold:
                labels.append(-1)
            else:
                labels.append(0)
            yesterday = today[3]
        final_labels = labels[time_batch:]

    return final_labels


def create_crossentropy_labels(stocks, time_batch, up_threshold, down_threshold):

    labels = []
    for stock in enumerate(stocks):
        cnd = stock.data[0]
        yesterday = cnd[0][3]
        for i, today in enumerate(cnd):
            if today[3] > yesterday * up_threshold:
                labels.append([1, 0, 0])
            elif today[3] < yesterday * down_threshold:
                labels.append([0, 1, 0])
            else:
                labels.append([1, 0, 1])
            yesterday = today[3]
        final_labels = labels[time_batch:]

    return final_labels


def min_max_normalization(data):

    dmax = np.amax(np.amax(data, axis=0), axis=1, keepdims=True)
    dmin = np.amax(np.amax(data, axis=0), axis=1, keepdims=True)
    data = np.divide(data-dmin, dmax-dmin)
    return data

x, y = model_fitting_data(3000, time_batch=16, s_list=['a', 'aa', 'blbbl'], f_list=None, labels_type='tanh')
