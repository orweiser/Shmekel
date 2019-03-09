import numpy as np
from ShmekelCore import Feature


class AD(Feature):
    def __init__(self, **kwargs):
        """
        use this method to define the parameters of the feature
        """
        super(AD, self).__init__(**kwargs)

    def _compute_feature(self, data):
        """
        A/D (Accumulation/Distribution) indicator is a momentum indicator that attempts to gauge supply and demand
        data columns represent'Open', 'High', 'Low', 'Close', 'Volume'
                                0       1       2       3         4
        """
        close = self._get_basic_feature(data[0], 'close')
        low = self._get_basic_feature(data[0], 'low')
        high = self._get_basic_feature(data[0], 'high')
        volume = self._get_basic_feature(data[0], 'volume')
        # M = data[0].shape[0]
        # AD = np.zeros(M)
        ad = np.multiply(np.divide(2 * close - low - high, high - low), volume)
        ad = np.cumsum(ad)
        return ad
