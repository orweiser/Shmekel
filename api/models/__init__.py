from .fully_connected import FullyConnected
from .lstm import LSTM
from .LSTM_DropOut import LSTM_DropOut
from .lstm_compose import LSTM_compose
from .general_RNN import GeneralRnn
from .none_sequential_architecture import NoneSequentialArchitecture


def get(model: str, **kwargs):
    if model == 'FullyConnected':
        return FullyConnected(model=model, **kwargs)
    if model == 'LSTM':
        return LSTM(model=model, **kwargs)
    if model == 'LSTM_DropOut':
        return LSTM_DropOut(model=model, **kwargs)
    if model == 'LSTM_compose':
        return LSTM_compose(model=model, **kwargs)
    if model == 'General_RNN':
        return GeneralRnn(model=model, **kwargs)
    if model == 'non_sequential_architecture':
        return NoneSequentialArchitecture(model=model, **kwargs)
    raise ValueError('Unexpected model, got ' + model)

