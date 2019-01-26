import numpy as np
from ShmekelCore import Feature, Math


class CCI(Feature):

    def __init__(self, period=14, **kwargs):
        """
        use this method to define the parameters of the feature
        """
        super(CCI, self).__init__(**kwargs)

        self.time_delay = period - 1

        self._period = period

    def _compute_feature(self, data):
        close = self._get_basic_feature(data[0], 'close')
        low = self._get_basic_feature(data[0], 'low')
        high = self._get_basic_feature(data[0], 'high')

        typical_price = (high + low + close) / 3
        sma = Math.smooth_moving_avg(typical_price, self._period)
        mean_abs_val = np.mean(np.abs(typical_price - np.mean(typical_price)))

        cci = (typical_price[:np.size(sma)] - sma)/(0.015*mean_abs_val)

        return cci
