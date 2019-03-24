import numpy as np
from shmekel_core import Feature, math


class RSI(Feature):
    def __init__(self, period=14, **kwargs):
        """
        use this method to define the parameters of the feature
        """
        super(RSI, self).__init__(**kwargs)

        self.time_delay = period
        self.is_numerical = True

    def _compute_feature(self, data):
        close = self._get_basic_feature(data[0], 'close')
        dif = -np.diff(close)
        u = np.maximum(dif, 0)
        d = np.maximum(-dif, 0)

        smmau = math.smooth_moving_avg(u, self.time_delay)
        smmad = math.smooth_moving_avg(d, self.time_delay)

        rs = smmau / (smmad + math.EPS_DOUBLE)  # adding epsilon for numerical purposes
        rsi = 100 - 100 / (1 + rs)
        return rsi
