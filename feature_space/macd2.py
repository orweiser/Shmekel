import numpy as np
from shmekel_core import Feature
from shmekel_core.math import exponential_moving_average


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

        self.time_delay = 26
        if calc_signal_line:
            self.time_delay += 9

    def _compute_feature(self, data, feature_list=None):

        close = self._get_basic_feature(data[0], 'close')
        low = self._get_basic_feature(data[0], 'low')
        high = self._get_basic_feature(data[0], 'high')
        return self.process(high, low, close)

    def process(self, high, low, close): #  Should we use typical price or other candle components (didn't find instructions online)

        typical_price = (high + low + close) / 3.
        ema_12 = exponential_moving_average(typical_price, 12)
        ema_26 = exponential_moving_average(typical_price, 26)

        macd = ema_12[:14] - ema_26
        if self._calc_signal_line:
            signal_line = exponential_moving_average(macd, 9)
            macd_dive_conv = macd[:9] - signal_line
            return macd_dive_conv
        return macd
