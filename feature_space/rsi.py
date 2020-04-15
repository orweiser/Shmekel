import numpy as np
from shmekel_core import Feature, math


class RSI(Feature):
    def __init__(self, range=14, normalization_type='default'):

        super(RSI, self).__init__(normalization_type=normalization_type, is_numerical=True,
                                  time_delay=range, num_features=1)
        self.range = range

    def _compute_feature(self, data, feature_list=None):
        close = self._get_basic_feature(data[0], 'close')
        return self.process(close)

    def process(self, close):
        dif = -np.diff(close)
        u = np.maximum(dif, 0)
        d = np.maximum(-dif, 0)

        smmau = math.smooth_moving_avg(u, self.range)
        smmad = math.smooth_moving_avg(d, self.range)

        rs = smmau / (smmad + math.EPS_DOUBLE)  # adding epsilon for numerical purposes
        rsi = 100 - 100 / (1 + rs)

        return rsi


class RSIOld(Feature):
    def __init__(self, period=14, **kwargs):
        """
        use this method to define the parameters of the feature
        """
        super(RSIOld, self).__init__(**kwargs)

        self.time_delay = period
        self.is_numerical = True

    def _compute_feature(self, data, feature_list=None):
        close = self._get_basic_feature(data[0], 'close')
        dif = -np.diff(close)
        u = np.maximum(dif, 0)
        d = np.maximum(-dif, 0)

        smmau = math.smooth_moving_avg(u, self.time_delay)
        smmad = math.smooth_moving_avg(d, self.time_delay)

        rs = smmau / (smmad + math.EPS_DOUBLE)  # adding epsilon for numerical purposes
        rsi = 100 - 100 / (1 + rs)
        return rsi
