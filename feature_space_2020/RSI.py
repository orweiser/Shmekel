import numpy as np
from shmekel_core import Feature, math


class RSI(Feature):
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
        dif = -np.diff(close)
        u = np.maximum(dif, 0)
        d = np.maximum(-dif, 0)

        smmau = math.smooth_moving_avg(u, self.range)
        smmad = math.smooth_moving_avg(d, self.range)

        rs = smmau / (smmad + math.EPS_DOUBLE)  # adding epsilon for numerical purposes
        rsi = 100 - 100 / (1 + rs)
        if self.auto_fill:
            delta = len(close)-len(rsi)
            fill = np.full(delta, rsi[0])
            rsi = np.concatenate([fill, rsi])
        return rsi
