import numpy as np
from ShmekelCore import Feature

class SMA(Feature):
    def __init__(self,range = 14, data=None, normalization_type=None, time_delay=0, is_numerical=None):
        self.range = range
        self.data = data
        self.normalization_type = normalization_type
        self.is_numerical = 1
        self.time_delay = time_delay

    def _compute_feature(self, data):
        close = self._get_basic_feature(data[0], 'close')
        wArray = np.arange(0, self.range) + 1
        weights = wArray / wArray.sum()
        return np.convolve(close, weights[::-1], mode='valid')
