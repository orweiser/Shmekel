from .mnist import MNIST
from .stock_dataset import StocksDataSet


def get(dataset: str, **kwargs):
    if dataset == 'MNIST':
        return MNIST(dataset=dataset, **kwargs)

    elif dataset == 'StocksDataSet':
        return StocksDataSet(dataset=dataset, **kwargs)

    raise ValueError('Unexpected dataset, got ' + dataset)
