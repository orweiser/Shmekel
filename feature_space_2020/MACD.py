import numpy as np
from shmekel_core import Feature
from shmekel_core.math import exponential_moving_average, exponential_moving_average_investopedia


class MACD(Feature):
    def __init__(self, calc_signal_line=False, normalization_type='default'):
        """
        MACD is defined as : EMA_12(data) - EMA_26(data),  where EMA_x(data) is the x period EMA calculated on data.
        The signal line is the EMA_9(MACD) and is usually used as a trigger for buy and sell signals
        :param calc_signal_line: if true will add column of the EMA_9(MACD) on the right. This will also add 9 to the
        time_delay of the feature.
        :type calc_signal_line: bool
        """
        super(MACD, self).__init__(normalization_type=normalization_type, is_numerical=True)
        self._calc_signal_line = calc_signal_line

        self.range1 = 12
        self.range2 = 26
        self.range_signal = 9
        self.time_delay = self.range2
        self.num_features = 1
        if calc_signal_line:
            self.time_delay += self.range_signal
            self.num_features = 2

        self.auto_fill = True

    def _compute_feature(self, data, feature_list=None):

        close = self._get_basic_feature(data[0], 'close')
        return self.process(close)

    def process(self, close): #  Should we use typical price or other candle components (didn't find instructions online)

        range1, range2, range_s = self.range1, self.range2, self.range_signal
        ema_12 = exponential_moving_average_investopedia(close, range1)
        ema_26 = exponential_moving_average_investopedia(close, range2)
        macd = ema_12[range2 - range1:] - ema_26

        if self._calc_signal_line:
            signal_line = exponential_moving_average_investopedia(macd, range_s)
            macd_plus_signal = np.concatenate((np.expand_dims(macd[range_s-1:], 1), np.expand_dims(signal_line, 1)), axis=1)
            if self.auto_fill:
                delta = len(close) - len(signal_line)
                fill = np.transpose([np.full(delta, macd_plus_signal[0, 0]), np.full(delta, macd_plus_signal[0, 1])])
                macd_plus_signal = np.concatenate([fill, macd_plus_signal])
            return macd_plus_signal

        if self.auto_fill:
            delta = len(close) - len(macd)
            fill = np.full(delta, macd[0])
            macd = np.concatenate([fill, macd])

        return macd
