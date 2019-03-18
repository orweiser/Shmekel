import numpy as np
from shmekel_core import Feature


class VWAP(Feature):
    def __init__(self, period=14, **kwargs):
        """
        use this method to define the parameters of the feature
        """
        super(VWAP, self).__init__(**kwargs)
        self.time_delay = period

    def _compute_feature(self, data):
        """
        VWAP (Volume Weighted Average Price)
        data columns represent'Open', 'High', 'Low', 'Close', 'Volume'
                                0       1       2       3         4
        """
        close = self._get_basic_feature(data[0], 'close')
        volume = self._get_basic_feature(data[0], 'volume')

        N = self.time_delay
        M = data[0].shape[0]
        vwap = np.zeros(M - N - 1)
        vc = np.multiply(close, volume)
        for i in range(M - N - 1):
            vwap[i] = np.divide(sum(vc[i:i + N]), sum(volume[i:i + N]))
        return vwap
