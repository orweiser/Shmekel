import numpy as np
from ShmekelCore import Feature

class PPO(Feature):
    def __init__(self,fast_range=5,slow_range=10 , **kwargs):
        """
        use this method to define the parameters of the feature
        """
        super(PPO, self).__init__(**kwargs)

        self.time_delay = slow_range
        self.is_numerical = True

    def getSma(self,icolumn=None,irange=None):
        cumsum = np.cumsum(np.insert(icolumn, 0, 0))
        result = (cumsum[irange:] - cumsum[:-irange]) / float(irange)
        result = np.concatenate((icolumn[:irange-1], result))
        return result

    def _compute_feature(self, data):
        close = self._get_basic_feature(data[0], 'close')
        fast_sma = self.getSma(close, self.fastrange)
        slow_sma = self.getSma(close, self.slowrange)
        ppo = ((fast_sma - slow_sma) / slow_sma) * 100
        return np.array(ppo[self.slowrange - 1:])

