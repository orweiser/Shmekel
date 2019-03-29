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
        range = self.range
        cumsum = np.cumsum(np.insert(close, 0, 0))
        return (cumsum[range:] - cumsum[:-range]) / float(range)
