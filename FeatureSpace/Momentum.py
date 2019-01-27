import numpy as np
from ShmekelCore import Feature


class Momentum(Feature):
    def __init__(self, period=14, **kwargs):
        """
        use this method to define the parameters of the feature
        """
        super(Momentum, self).__init__(**kwargs)

        self.time_delay = period

    def _compute_feature(self, data):
        """
         momentum indicator compares the current price to selected number of previous prices.
         The function recieve Closer values and return each day (current close value) - sum(previous close values)
         data columns represent'Open', 'High', 'Low', 'Close', 'Volume'
                                0       1       2       3         4
        """
        close = self._get_basic_feature(data[0], 'close')
        N = self.time_delay
        M = data[0].shape[0]
        mom = np.zeros(M - N - 1)
        for i in range(M - N - 1):
            mom[i] = close[i + N] - sum(close[i:i + N])
        return mom
