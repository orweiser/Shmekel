import numpy as np
from shmekel_core import Feature, math


class ADX(Feature):

    def __init__(self, period=14, **kwargs):
        """
        use this method to define the parameters of the feature
        """
        super(ADX, self).__init__(**kwargs)

        self.time_delay = 2*period - 1
        self._period = period

    def _compute_feature(self, data, feature_list=None):
        close = self._get_basic_feature(data[0], 'close')
        low = self._get_basic_feature(data[0], 'low')
        high = self._get_basic_feature(data[0], 'high')

        up_move = -np.diff(high)
        down_move = np.diff(low)

        true_range = np.amax(np.array([high[:-1] - low[:-1], np.abs(high[:-1] - close[1:]),
                                       np.abs(low[:-1] - close[1:])]), axis=0)
        avg_true_range = math.smooth_moving_avg(true_range, self._period)

        dm_plus = up_move * (up_move > 0) * (up_move > down_move) / avg_true_range
        dm_minus = down_move * (down_move > 0) * (down_move > up_move) / avg_true_range

        di_plus = math.smooth_moving_avg(dm_plus, self._period)
        di_minus = math.smooth_moving_avg(dm_minus, self._period)
        adx = 100 * math.smooth_moving_avg(np.abs((di_plus - di_minus) / (di_plus + di_minus)), self._period)

        return adx
