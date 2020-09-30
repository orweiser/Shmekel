from .ad import AD
from .adl import ADL
from .adx import ADX
from .basic_features import *
from .output_features import *
from .cci import CCI
from .derivatives import Derivatives
from .macd import MACD
from .mfi import MFI
from .momentum import Momentum
from .vwap import VWAP
from .time_ratio import TimeRatio
from .stochastic import Stochastic
from .rsi import RSI
from .sma import SMA
from Utils.logger import logger


mapping = {
    'high': High,
    'open': Open,
    'low': Low,
    'close': Close,
    'volume': Volume,
    'candle': Candle,
    'datetuple': DateTuple,
    'rawcandle': RawCandle,
    'rise': Rise,
    'ad': AD,
    'adl': ADL,
    # 'adx': ADX,  # todo: uncomment when adx is fixed
    'cci': CCI,
    'derivatives': Derivatives,
    'macd': MACD,
    'mfi': MFI,
    'momentum': Momentum,
    'vwap': VWAP,
    'time_ratio': TimeRatio,
    'rsi': RSI,
    'sma': SMA,
    # 'stochastic': Stochastic,  # todo: uncomment when it is fixed
    'hor': HighOpenRatio,
    'cor': CloseOpenRatio,
}


def get_feature(feature_name, **params):
    assert isinstance(feature_name, str)
    logger.debug('Fetching %s Feature', feature_name)
    return mapping[feature_name.lower()](**params)
