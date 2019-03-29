import numpy as np
import pandas as pd
from ShmekelCore import Feature

class SMA(Feature):
    def __init__(self,range = 14, data=None, normalization_type=None, time_delay=0, is_numerical=None):
        self.range = range
        self.data = data
        self.normalization_type = normalization_type
        self.is_numerical = 1
        self.time_delay = time_delay

    def _compute_feature(self, data):
        close = self._get_basic_feature(data[0], 'close')
        df = pd.DataFrame(close)
        ema_short = df.ewm(span=self.range, adjust=False).mean()
        rtrn = np.array(ema_short.values)[self.range - 1:]
        return rtrn.reshape(1, rtrn.shape[0])
