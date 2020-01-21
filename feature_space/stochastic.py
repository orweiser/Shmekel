from shmekel_core import Feature
import numpy as np


class Stochastic(Feature):
    def __init__(self, period=14, **kwargs):  # DO NOT CHANGE THE DECLARATION
        """
        use this method to define the parameters of the feature
        """
        super(Stochastic, self).__init__(**kwargs)

        self.time_delay = period + 2

        self._period = period

    def _compute_feature(self, data, feature_list=None):
        close = self._get_basic_feature(data[0], 'close')
        low = self._get_basic_feature(data[0], 'low')
        high = self._get_basic_feature(data[0], 'high')

        K =  np.zeros(np.size(close) - self._period + 1)
        for idx in range(np.size(close) - self._period):

            K[idx] = 100*(close[idx] - np.max(low[idx:(idx + self._period)]))/(np.max(high[idx:(idx + self._period)])
                                                                               - np.max(low[idx:(idx + self._period)]))
        return K[0], (K[0] + K[1] + K[2])/3