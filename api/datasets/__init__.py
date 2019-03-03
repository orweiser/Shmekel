from .mnist import MNIST


def get(dataset: str, **kwargs):
    if dataset == 'MNIST':
        return MNIST(dataset=dataset, **kwargs)

    raise ValueError('Unexpected dataset, got ' + dataset)
