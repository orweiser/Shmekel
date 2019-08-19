from .fully_connected import FullyConnected
from .lstm import LSTM
from utils.logger import logger


@logger.info_dec
def get(model: str, **kwargs):
    if model == 'FullyConnected':
        return FullyConnected(model=model, **kwargs)
    if model == 'LSTM':
        return LSTM(model=model, **kwargs)

    raise ValueError('Unexpected model, got ' + model)

