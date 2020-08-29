import numpy as np
from shmekel_core import Feature, math


class CCI(Feature):
    def __init__(self, range=14, normalization_type='default'):
        super(CCI, self).__init__(normalization_type=normalization_type, is_numerical=True, time_delay=range)

        self.range = range
        self.auto_fill = True

    def _compute_feature(self, data, feature_list=None):
        high = self._get_basic_feature(data[0], 'high')
        low = self._get_basic_feature(data[0], 'low')
        close = self._get_basic_feature(data[0], 'close')

        return self.process(high, low, close)

    def process(self, high, low, close):
        typical_price = (high + low + close) / 3
        ma = math.smooth_moving_avg_investopedia(typical_price, self.range)
        ma_size = np.size(ma)
        deviation = np.zeros((ma_size))
        for i in range(ma_size):
            ma_abs_val = np.abs(typical_price[i: i+self.range] - ma[i])
            deviation[i] = np.sum(ma_abs_val) / self.range  # range - 1 might be more suitable here

        cci = (typical_price[self.range-1:] - ma) / (0.015 * deviation)

        if self.auto_fill:
            delta = len(typical_price)-len(cci)
            fill = np.full(delta, cci[0])
            cci = np.concatenate([fill, cci])

        return cci
