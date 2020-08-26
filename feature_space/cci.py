import numpy as np
from shmekel_core import Feature, math


class CCI(Feature):
    def __init__(self, range=14, normalization_type='default'):
        super(CCI, self).__init__(normalization_type=normalization_type, is_numerical=True, time_delay=range)

        self.range = range

    def _compute_feature(self, data, feature_list=None):
        high = self._get_basic_feature(data[0], 'high')
        low = self._get_basic_feature(data[0], 'low')
        close = self._get_basic_feature(data[0], 'close')

        return self.process(high, low, close)

    def process(self, high, low, close):
        typical_price = (high + low + close) / 3
        ma = math.smooth_moving_avg(typical_price, self.range)
        ma_size = np.size(ma)
        ma_abs_val = math.smooth_moving_avg(np.abs(typical_price[ma_size] - ma), self.range)
        deviation = np.divide(ma_abs_val, self.range)  # period - 1 might be more suitable here

        cci = (typical_price[:ma_size] - ma)/(0.015*deviation)

        return cci
