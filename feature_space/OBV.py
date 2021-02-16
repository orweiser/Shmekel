import numpy as np
from ShmekelCore import Feature

class OBV(Feature):
    def __init__(self, **kwargs):
        """
        use this method to define the parameters of the feature
        """
        super(OBV, self).__init__(**kwargs)

        self.time_delay = 0
        self.is_numerical = True

    def _compute_feature(self, data):
        close = self._get_basic_feature(data[0], 'close')
        volume = self._get_basic_feature(data[0], 'volume')
        obv = []
        curr_obv = 0
        obv.append(curr_obv)
        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                curr_obv = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                curr_obv = obv[i - 1] - volume[i]
            else:
                curr_obv = obv[i - 1]
            obv.append(curr_obv)

        return np.array(obv)

