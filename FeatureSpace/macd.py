import numpy as np
from ShmekelCore import Feature
from ShmekelCore.math import exponential_moving_average


class MACD(Feature):
    def __init__(self, calc_signal_line=False, **kwargs):
        """
        MACD is defined as : EMA_12(data) - EMA_26(data),  where EMA_x(data) is the x period EMA calculated on data.
        The signal line is the EMA_9(MACD) and is usually used as a trigger for buy and sell signals
        :param calc_signal_line: if true will add column of the EMA_9(MACD) on the right. This will also add 9 to the
        time_delay of the feature.
        :type calc_signal_line: bool
        """
        super(MACD, self).__init__(**kwargs)
        self._calc_signal_line = calc_signal_line

        self.time_delay = 26
        if calc_signal_line:
            self.time_delay += 9

    def _compute_feature(self, data):
        close = self._get_basic_feature(data[0], 'close')
        low = self._get_basic_feature(data[0], 'low')
        high = self._get_basic_feature(data[0], 'high')

        price = (high + low + close) / 3.
        ema_12 = exponential_moving_average(price, 12)
        ema_26 = exponential_moving_average(price, 26)
        macd = ema_12[:14] - ema_26
        if self._calc_signal_line:
            signal_line = exponential_moving_average(macd, 9)
            return np.concatenate((macd, signal_line), axis=1)
        return macd
