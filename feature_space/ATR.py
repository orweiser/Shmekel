import numpy as np
from ShmekelCore import Feature


class ATR(Feature):
    def __init__(self, period=10, **kwargs):
        """
        use this method to define the parameters of the feature
        """
        super(ATR, self).__init__(**kwargs)
        self.range = period
        self.time_delay = period
        self.is_numerical = True

    def getSma(self, icolumn=None, irange=None):
        cumsum = np.cumsum(np.insert(icolumn, 0, 0))
        result = (cumsum[irange:] - cumsum[:-irange]) / float(irange)
        result = np.concatenate((icolumn[:irange - 1], result))
        return result

    def _compute_feature(self, data):
        high = self._get_basic_feature(data[0], 'high')
        low = self._get_basic_feature(data[0], 'low')
        high_minus_low = high - low
        atr = self.getSma(high_minus_low, self.range)
        return np.array(atr[self.range - 1:])
