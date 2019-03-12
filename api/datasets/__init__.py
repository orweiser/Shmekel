from .mnist import MNIST
from .data_generator import Generator


def get(dataset: str, **kwargs):
    if dataset == 'MNIST':
        return MNIST(dataset=dataset, **kwargs)
    else:
        print('kwargs: {}'.format(kwargs))
        return Generator(dataset=dataset, **kwargs)

    # raise ValueError('Unexpected dataset, got ' + dataset)
