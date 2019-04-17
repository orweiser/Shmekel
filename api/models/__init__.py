from .fully_connected import FullyConnected
from .lstm import LSTM
from .LSTM_DropOut import LSTM_DropOut
from .composing_model import LSTM_compose


def get(model: str, **kwargs):
    if model == 'FullyConnected':
        return FullyConnected(model=model, **kwargs)
    if model == 'LSTM':
        return LSTM(model=model, **kwargs)
    if model == 'LSTM_DropOut':
        return LSTM_DropOut(model=model, **kwargs)
    if model == 'LSTM_compose':
        return LSTM_compose(model=model, **kwargs)
    raise ValueError('Unexpected model, got ' + model)

