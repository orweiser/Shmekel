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
        dif = np.diff(close) / close[:-1]
        up = np.maximum(dif, 0)
        down = -np.minimum(dif, 0)

        avg_up = self.rsi_sma(up)
        avg_down = self.rsi_sma(down)

        rs = avg_up / (avg_down + math.EPS_DOUBLE)  # adding epsilon for numerical purposes
        rsi = 100 - 100 / (1 + rs)
        # if self.auto_fill:
        #     delta = len(close) - len(rsi)
        #     fill = np.full(delta, rsi[0])
        #     rsi = np.concatenate([fill, rsi])
        return rsi

    def rsi_sma(self, x):
        smma_out = []
        init_avg = np.average(x[0:self.range])
        smma_out.append(init_avg)
        multiplyer = self.range - 1
        for xval in x[self.range:]:
            last_val = smma_out[-1]
            new_val = (multiplyer * last_val + xval) / self.range
            smma_out.append(new_val)
            # print(new_val)
        return np.asarray(smma_out)


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
