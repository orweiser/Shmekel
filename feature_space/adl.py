import numpy as np
from shmekel_core import Feature, math


class ADL(Feature):
    def __init__(self, **kwargs):
        """
        use this method to define the parameters of the feature
        """
        super(ADL, self).__init__(**kwargs)

    def _compute_feature(self, data, feature_list=None):
        close = self._get_basic_feature(data[0], 'close')
        low = self._get_basic_feature(data[0], 'low')
        high = self._get_basic_feature(data[0], 'high')
        volume = self._get_basic_feature(data[0], 'volume')

        AD = volume*(2 * close - low - high)/(high - low + math.EPS_DOUBLE)
        # AD = volume*(2 * close + (-low) + (-high))/(high + (-low))

        return AD  # AD[0]
