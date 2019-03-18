from .ad import AD
from .adl import ADL
from .adx import ADX
from .basic_features import *


def get_feature(feature_name, **params):
    assert isinstance(feature_name, str)

    if feature_name.lower() == 'ad': return AD(**params)
    elif feature_name.lower() == 'adl': return ADL(**params)
    elif feature_name.lower() == 'adx': return ADX(**params)
    elif feature_name.lower() == 'candle': return Candle(**params)
    elif feature_name.lower() == 'high': return High(**params)
    elif feature_name.lower() == 'open': return Open(**params)
    elif feature_name.lower() == 'close': return Close(**params)
    elif feature_name.lower() == 'low': return Low(**params)

    raise KeyError('unexpected feature_name. got: ' + feature_name)

