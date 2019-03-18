import numpy as np
from shmekel_core import Feature


class TimeRatio(Feature):
    def __init__(self, **kwargs):
        """
        This feature is the ratio between the sample from this time stamp and the sample of the previous time stamp:
        X[n] / X[n-1], in our case X[n] is the entire candle (i.e. 5 element vector)
        """
        super(TimeRatio, self).__init__(**kwargs)
        self.time_delay = 1
        self.num_features = 5 # performed on the entire candle

    def _compute_feature(self, data):
        candle = np.zeros((len(data[0]), 5))
        candle[:, 0] = self._get_basic_feature(data[0], 'open')
        candle[:, 1] = self._get_basic_feature(data[0], 'close')
        candle[:, 2] = self._get_basic_feature(data[0], 'high')
        candle[:, 3] = self._get_basic_feature(data[0], 'low')
        candle[:, 4] = self._get_basic_feature(data[0], 'volume')

        return np.divide(candle[:-1, :], candle[1:,:])
