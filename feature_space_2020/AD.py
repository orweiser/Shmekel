import numpy as np
from shmekel_core import Feature

# TODO: not complite
class AD(Feature):
    def __init__(self, period=None, data=None, normalization_type=None, time_delay=0):
        # self.range = period
        self.data = data
        self.normalization_type = normalization_type
        self.time_delay = time_delay
        self.is_numerical = True
        self.auto_fill = True

    def _compute_feature(self, data):
        close = self._get_basic_feature(data[0], 'close')
        high = self._get_basic_feature(data[0], 'high')
        low = self._get_basic_feature(data[0], 'low')
        volume = self._get_basic_feature(data[0], 'volume')
        return self.process(close, high, low, volume)

    def process(self, close, high, low, volume):
        processed = (close - low) - (high - close)
        processed = np.divide(processed, high-low)
        processed = np.multiply(processed, volume)
        processed = np.cumsum
        if self.auto_fill:
            delta = len(close)-len(processed)
            fill = np.full(delta,processed[0])
            processed = np.concatenate([fill, processed])
        return processed
