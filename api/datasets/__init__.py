from .mnist import MNIST
from .stock_dataset import StocksDataset


def get(dataset: str, **kwargs):
    if dataset == 'MNIST':
        return MNIST(dataset=dataset, **kwargs)

    elif dataset == 'StocksDataset':
        return StocksDataset(dataset=dataset, **kwargs)

    raise ValueError('Unexpected dataset, got ' + dataset)
