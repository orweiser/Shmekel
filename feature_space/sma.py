import numpy as np
from shmekel_core import Feature


class SMA(Feature):
    def __init__(self, range=14, data=None, normalization_type='default'):
        super(SMA, self).__init__(normalization_type=normalization_type, time_delay=range)

        self.range = range
        self.data = data
        self.normalization_type = normalization_type
        self.is_numerical = True

    def _compute_feature(self, data, feature_list=None):
        close = self._get_basic_feature(data[0], 'close')
        return self.process(close)

    def process(self, close):
        range = self.range
        # todo: figure out if the "insert" is needed
        # cumsum = np.cumsum(np.insert(close, 0, 0))
        cumsum = np.cumsum(close)
        processed = (cumsum[range:] - cumsum[:-range]) / float(range)
        return processed
