from Dstruct import *

import numpy as np


def get_base_identifier(data, key):
    """
    :type key: str

    :param data: TBD
    :param key: oner of the following: 'low', 'high', 'open', 'close', 'volume' or 'date'
    :return: numpy array if not 'date', else list
    """
    return np.array([1])


class SMMA(Feature):
    def __init__(self, time_series=None, period=14, data=None, normalization_type=None):  # DO NOT CHANGE THE DECLARATION
        """
        use this method to define the parameters of the feature
        """

        self.time_delay = period - 1  # change it according to the feature as described in class Feature
        self.is_numerical = 1  # boolean. change it according to the feature as described in class Feature

        # here you can define more parameters that "_compute_feature" might need to use
        self.time_series = time_series
        self.period = period

        # the following line must be included
        super(SMMA, self).__init__(data=data, normalization_type=normalization_type)

    def _compute_feature(self, data):
        serie = np.flip(self.time_series)
        smma = np.ndarray(np.size(self.time_series) - self.time_delay + 1)
        smma[0] = np.mean(serie[None:self.period])

        for idx in range(self.period, np.size(serie) + 1):
            smma[idx - self.period + 1] = (1 - 1/self.period)*smma[idx - self.period] + (1/self.period)*serie[idx]

        smma = np.flip(smma)
        return smma


class RSI(Feature):
    def __init__(self, period=14, data=None, normalization_type=None):  # DO NOT CHANGE THE DECLARATION
        """
        use this method to define the parameters of the feature
        """

        self.time_delay = period + 1  # change it according to the feature as described in class Feature
        self.is_numerical = 1  # boolean. change it according to the feature as described in class Feature

        # here you can define more parameters that "_compute_feature" might need to use
        self.period = period
        self.epsilon = 1e-8

        # the following line must be included
        super(RSI, self).__init__(data=data, normalization_type=normalization_type)

    def _compute_feature(self, data):
        close = get_base_identifier(data, 'close')
        close = np.flip(close)
        dif = np.diff(close)
        U = np.maximum(dif, 0)
        D = np.maximum(-dif, 0)

        SMMAU = np.mean(U[None:self.period])
        SMMAD = np.mean(D[None:self.period])

        for idx in range(self.period, np.size(U)+1):
            SMMAU = np.append(SMMAU, (1 - 1/self.period)*SMMAU[-1] + (1/self.period)*U[idx])
            SMMAD = np.append(SMMAD, (1 - 1/self.period)*SMMAD[-1] + (1/self.period)*D[idx])

        rs = SMMAU / (SMMAD + self.epsilon)
        rsi = 100 - 100 / (1 + rs)
        rsi = np.flip(rsi)
        return rsi


class ADL(Feature):
    def __init__(self, data=None, normalization_type=None):  # DO NOT CHANGE THE DECLARATION
        """
        use this method to define the parameters of the feature
        """

        self.time_delay = 0  # change it according to the feature as described in class Feature
        self.is_numerical = 1  # boolean. change it according to the feature as described in class Feature

        # here you can define more parameters that "_compute_feature" might need to use

        # the following line must be included
        super(ADL, self).__init__(data=data, normalization_type=normalization_type)

    def _compute_feature(self, data):
        close = get_base_identifier(data, 'close')
        low = get_base_identifier(data, 'low')
        high = get_base_identifier(data, 'high')
        volume = get_base_identifier(data, 'volume')

        AD = volume*(2 * close - low - high)/(high - low)
        # AD = volume*(2 * close + (-low) + (-high))/(high + (-low))

        return AD[0]


class MFI(Feature):
    def __init__(self, days, data=None, normalization_type=None):  # DO NOT CHANGE THE DECLARATION
        """
        use this method to define the parameters of the feature
        """

        self.time_delay = days  # change it according to the feature as described in class Feature
        self.is_numerical = 1  # boolean. change it according to the feature as described in class Feature

        # here you can define more parameters that "_compute_feature" might need to use
        self.days = days
        # the following line must be included
        super(MFI, self).__init__(data=data, normalization_type=normalization_type)

    def _compute_feature(self, data):
        close = get_base_identifier(data, 'close')
        low = get_base_identifier(data, 'low')
        high = get_base_identifier(data, 'high')
        volume = get_base_identifier(data, 'volume')

        typical_price = ( high + low + close)/3
        money_flow = volume*typical_price

        positive_money_flow = np.zeros(1)
        negative_money_flow = np.zeros(1)

        for idx in range(self.days):

            if typical_price[idx] > typical_price[idx+1]:
                positive_money_flow += money_flow[idx]

            if typical_price[idx] < typical_price[idx+1]:
                negative_money_flow += money_flow[idx]

        money_ration = positive_money_flow/negative_money_flow
        money_flow_index = 100 - 100/(1 + money_ration)

        return money_flow_index


class Stochastic(Feature):
    def __init__(self, period=14, data=None, normalization_type=None):  # DO NOT CHANGE THE DECLARATION
        """
        use this method to define the parameters of the feature
        """

        self.time_delay = period + 2  # change it according to the feature as described in class Feature
        self.is_numerical = 1  # boolean. change it according to the feature as described in class Feature

        # here you can define more parameters that "_compute_feature" might need to use
        self.period = period
        # the following line must be included
        super(Stochastic, self).__init__(data=data, normalization_type=normalization_type)

    def _compute_feature(self, data):
        close = get_base_identifier(data, 'close')
        low = get_base_identifier(data, 'low')
        high = get_base_identifier(data, 'high')

        K =  np.zeros(np.size(close) - self.period + 1)
        for idx in range(np.size(close) - self.period):

            K[idx] = 100*(close[idx] - np.max(low[idx:(idx + self.period)]))/(np.max(high[idx:(idx + self.period)]) - np.max(low[idx:(idx + self.period)]))

        return K[0], (K[0] + K[1] + K[2])/3


class BollingerBands(Feature): # returns 2 degenerate fetures

    def __init__(self, smoothing_period=20, number_of_SD=2, data=None, normalization_type=None):  # DO NOT CHANGE THE DECLARATION
        """
        use this method to define the parameters of the feature
        """

        self.time_delay = smoothing_period  # change it according to the feature as described in class Feature
        self.is_numerical = 1  # boolean. change it according to the feature as described in class Feature

        # here you can define more parameters that "_compute_feature" might need to use
        self.smoothing_period = smoothing_period
        self.number_of_SD = number_of_SD
        # the following line must be included
        super(BollingerBands, self).__init__(data=data, normalization_type=normalization_type)

    def _compute_feature(self, data):
        close = get_base_identifier(data, 'close')
        low = get_base_identifier(data, 'low')
        high = get_base_identifier(data, 'high')

        typical_price = (high + low + close)/3
        moving_avg = np.ndarray(np.size(typical_price) - self.smoothing_period)
        std = np.ndarray(np.size(typical_price) - self.smoothing_period)


        for idx in range(0, np.size(typical_price) - self.smoothing_period):
            moving_avg[idx] = np.mean(typical_price[idx:(idx+self.smoothing_period)])
            std[idx] = np.std(typical_price[idx:(idx + self.smoothing_period)])

        return moving_avg + self.number_of_SD*std, moving_avg - self.number_of_SD*std


class ADX(Feature): # returns 2 degenerate fetures

    def __init__(self, period=14, data=None, normalization_type=None):  # DO NOT CHANGE THE DECLARATION
        """
        use this method to define the parameters of the feature
        """

        self.time_delay = period  # change it according to the feature as described in class Feature
        self.is_numerical = 1  # boolean. change it according to the feature as described in class Feature

        # here you can define more parameters that "_compute_feature" might need to use
        self.period = period
        # the following line must be included
        super(ADX, self).__init__(data=data, normalization_type=normalization_type)

    def _compute_feature(self, data):
        close = get_base_identifier(data, 'close')
        low = get_base_identifier(data, 'low')
        high = get_base_identifier(data, 'high')

        typical_price = (high + low + close)/3

        for idx in range(0, np.size(typical_price) - self.period):
            moving_avg = np.mean(typical_price[idx:(idx+self.period)])
            std = np.std(typical_price[idx:(idx + self.period)])

        return moving_avg + self.number_of_SD*std, moving_avg - self.number_of_SD*std


