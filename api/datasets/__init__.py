from .mnist import MNIST
from .data_generator import Generator
from .stock_dataset import StocksDataset
from .smooth_stocks import SmoothStocksDataset
from Utils.logger import logger


@logger.info_dec
def get(dataset: str, **kwargs):
    if dataset == 'MNIST':
        return MNIST(dataset=dataset, **kwargs)
    elif (dataset == 'Gen'):
        print('kwargs: {}'.format(kwargs))
        return Generator(dataset=dataset, **kwargs)
    elif dataset == 'StocksDataset':
        return StocksDataset(dataset=dataset, **kwargs)
    elif dataset == 'SmoothStocksDataset':
        return SmoothStocksDataset(dataset=dataset, **kwargs)

    raise ValueError('Unexpected dataset, got ' + dataset)
