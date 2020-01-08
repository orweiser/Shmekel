import numpy as np
from shmekel_core import Feature


class SMA(Feature):
    def __init__(self, period=14, data=None, normalization_type=None, time_delay=0):
        self.range = period
        self.data = data
        self.normalization_type = normalization_type
        self.time_delay = time_delay
        self.is_numerical = True
        self.auto_fill = True

    def _compute_feature(self, data):
        close = self._get_basic_feature(data[0], 'close')
        return self.process(close)

    def process(self, close):
        range = self.range
        cumsum = np.cumsum(np.insert(close, 0, 0))
        processed = (cumsum[range:] - cumsum[:-range]) / float(range)
        if self.auto_fill:
            delta = len(close)-len(processed)
            fill = np.full(delta,processed[0])
            processed = np.concatenate([fill, processed])
        return processed
