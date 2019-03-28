from shmekel_core import Feature
import numpy as np


# TODO: talk to Roee about this feature calculation seems wrong
class BollingerBands(Feature):  # returns 2 degenerate fetures

    def __init__(self, smoothing_period=20, number_of_SD=2, **kwargs):
        """
        use this method to define the parameters of the feature
        """
        super(BollingerBands, self).__init__(**kwargs)

        self.time_delay = smoothing_period

        self._smoothing_period = smoothing_period
        self._number_of_SD = number_of_SD

    def _compute_feature(self, data):
        close = self._get_basic_feature(data[0], 'close')
        low = self._get_basic_feature(data[0], 'low')
        high = self._get_basic_feature(data[0], 'high')

        typical_price = (high + low + close)/3
        moving_avg = np.ndarray(np.size(typical_price) - self._smoothing_period)
        std = np.ndarray(np.size(typical_price) - self._smoothing_period)

        for idx in range(0, np.size(typical_price) - self._smoothing_period):
            moving_avg[idx] = np.mean(typical_price[idx:(idx+self._smoothing_period)])
            std[idx] = np.std(typical_price[idx:(idx + self._smoothing_period)])

        return moving_avg + self._number_of_SD*std, moving_avg - self._number_of_SD*std
