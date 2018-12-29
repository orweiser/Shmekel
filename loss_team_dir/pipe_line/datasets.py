from keras.datasets import mnist, fashion_mnist, cifar10
import numpy as np
from keras.backend import epsilon


def load_dataset(dataset_name='mnist'):
    return {
        'mnist': mnist,
        'fashion': fashion_mnist, 'fashion_mnist': fashion_mnist,
        'cifar': cifar10, 'cifar10': cifar10,
    }[dataset_name].load_data()


def get_normalization_params(train_x, val_x=None):
    x = train_x if val_x is None else np.concatenate([train_x, val_x], axis=0)

    mu = np.mean(x, 0)
    std = np.std(x, 0)
    return mu, std


def normalize(train_x, val_x=None, normalization_params=None, epsilon=epsilon()):
    if not normalization_params:
        mu, std = get_normalization_params(train_x, val_x)
        if normalization_params is not None:
            (normalization_params['mu'], normalization_params['std']) = [mu, std]
    else:
        mu, sigma = [normalization_params['mu'], normalization_params['std']]

    def _normalize(x):
        return (x - mu) / (std + epsilon)

    if val_x is not None:
        return _normalize(train_x), _normalize(val_x)
    else:
        return _normalize(train_x)


def ind_generator(max_ind, randomize=True):
    while True:
        if randomize:
            ind_list = np.random.permutation(max_ind)
        else:
            ind_list = range(max_ind)

        for i in ind_list:
            yield i


def add_noise_to_sample(x, expectation=0, sigma=1):
    x += np.random.normal(expectation, sigma, x.shape)


def get_labels(classes, num_classes=10):
    I = np.eye(num_classes)
    return I[classes]


def _generator(x, batch_size, labels=None, randomize=True, noise_level=None):
    ind_gen = ind_generator(x.shape[0], randomize)
    batch_labels = None

    while True:
        batch_x = np.zeros((batch_size,) + x.shape[1:])
        if labels is not None:
            batch_labels = np.zeros((batch_size,) + labels.shape[1:])

        for batch_i, ind in enumerate(ind_gen):
            batch_x[batch_i] = x[ind]
            if noise_level is not None:
                add_noise_to_sample(batch_x[batch_i], sigma=noise_level)

            if labels is not None:
                batch_labels[batch_i] = labels[ind]

            if batch_i >= (batch_size - 1):
                break

        if labels is None:
            yield batch_x
        else:
            yield batch_x, batch_labels


def get_data_generators(dataset='mnist', with_labels=True, batch_size=128, noise_level=None, do_normalization=True,
                        normalization_params=None, num_classes=10, randomize=True):
    """
    creates two data generators - train and validation.
    :param dataset: dataset's name, one of: 'mnist', 'cifar10', 'fashion_mnist'
    :param with_labels: boolean. if true generators creates both input and output samples
    :param batch_size:
    :param noise_level: noise level to add to data
    :param do_normalization: boolean
    :param normalization_params: None or a dictionary
    :param num_classes: number of output classes
    :param randomize: boolean
    :return: a tuple of two generators: train, val
    """
    (train_x, train_y), (val_x, val_y) = load_dataset(dataset)

    if do_normalization:
        train_x, val_x = normalize(train_x, val_x, normalization_params)

    if with_labels:
        train_labels = get_labels(train_y, num_classes)
        val_labels = get_labels(val_y, num_classes)
    else:
        train_labels, val_labels = [None, None]

    train_gen = _generator(train_x, batch_size,
                           labels=None if not with_labels else train_labels,
                           randomize=randomize, noise_level=noise_level)
    val_gen = _generator(val_x, batch_size,
                         labels=None if not with_labels else val_labels,
                         randomize=randomize, noise_level=noise_level)

    return train_gen, val_gen

