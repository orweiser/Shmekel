from .fully_connected import FullyConnected


def get(model: str, **kwargs):
    if model == 'FullyConnected':
        return FullyConnected(**kwargs)

    raise ValueError('Unexpected model, got ' + model)

