import numpy as np
from shmekel_core import Feature


class MFI(Feature):
    def __init__(self, days, **kwargs):
        """
        use this method to define the parameters of the feature
        """
        super(MFI, self).__init__(**kwargs)

        self.time_delay = days

        self._days = days

    def _compute_feature(self, data, feature_list=None):
        close = self._get_basic_feature(data[0], 'close')
        low = self._get_basic_feature(data[0], 'low')
        high = self._get_basic_feature(data[0], 'high')
        volume = self._get_basic_feature(data[0], 'volume')

        typical_price = (high + low + close)/3
        money_flow = volume * typical_price

        # This look incorrect - TODO: Check with Roee
        positive_money_flow = np.zeros(1)
        negative_money_flow = np.zeros(1)

        for idx in range(self._days):

            if typical_price[idx] > typical_price[idx + 1]:
                positive_money_flow += money_flow[idx]

            if typical_price[idx] < typical_price[idx + 1]:
                negative_money_flow += money_flow[idx]

        money_ration = positive_money_flow / negative_money_flow
        money_flow_index = 100 - 100/(1 + money_ration)

        return money_flow_index
