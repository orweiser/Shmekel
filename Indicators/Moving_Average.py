from Dstruct import *
import numpy as np
import pandas as pd

class SMA(Feature):
    """
    get SMA (simple moving average) of a specific column in a numpy array
    by default: column number = 3 (for close column in data array )
    """

    def __init__(self,period=10,column_num = 3, data=None, normalization_type=None, time_delay=0, is_numerical=None):
        """

        :param period: range for average (default is 10)
        :param column_num: number of column to calculate  (default is column 3 for close prices)
        :param data:
        :param normalization_type:
        :param time_delay:
        :param is_numerical:
        """
        self.data = data
        self.normalization_type = normalization_type
        self.is_numerical = 1
        self.time_delay = time_delay

        self.range = period
        self.columnNum = column_num


    def __call__(self, *args, **kwargs):
        """Don't mind it, for later use"""
        self.__init__(*args, **kwargs)

    def _compute_feature(self, data):
        column = data[:, self.columnNum]
        range = self.range
        cumsum = np.cumsum(np.insert(column, 0, 0))
        return (cumsum[range:] - cumsum[:-range]) / float(range)

    def get_feature(self, temporal_delay=0, normalization_type=None):
        """
        compute feature on self.data
        ****This method should generally be inherited by children classes****
        :param temporal_delay: an integer.
            if temporal_delay and (temporal_delay < self.time_delay):
                raise Exception('while using method "get_feature" temporal_delay can not be smaller than self.time_delay)
            else:
                drop the oldest (temporal_delay - self.time_delay) values in  the feature
        :param normalization_type: normalization type as explained in the method "normalize".
        :return: a numpy array of shape (Stock_num_samples - time delay, feature size)
        """
        if temporal_delay and (temporal_delay < self.time_delay):
            raise Exception('while using method "get_feature" temporal_delay can not be smaller than self.time_delay')

        if self.data is None:
            return None

        feature = self._compute_feature(self.data)

        feature = normalize(feature, normalization_type=normalization_type)

        if temporal_delay > self.time_delay:
            feature = feature[:-(temporal_delay - self.time_delay)]

        return feature

class WMA(Feature):
    """
    get wMA ( weighted moving average) of a specific column in a numpy array
    by default: column number = 3 (for close column in data array )
    """

    def __init__(self,period=10,column_num = 3, data=None, normalization_type=None, time_delay=0, is_numerical=None):
        """

        :param period: range for average (default is 10)
        :param column_num: number of column to calculate  (default is column 3 for close prices)
        :param data:
        :param normalization_type:
        :param time_delay:
        :param is_numerical:
        """
        self.data = data
        self.normalization_type = normalization_type
        self.is_numerical = 1
        self.time_delay = time_delay

        self.range = period
        self.columnNum = column_num


    def __call__(self, *args, **kwargs):
        """Don't mind it, for later use"""
        self.__init__(*args, **kwargs)

    def _compute_feature(self, data):
        column = data[:, self.columnNum]
        wArray = np.arange(0, self.range)+1
        weights = wArray/wArray.sum()
        return np.convolve(column, weights[::-1], mode='valid')




    def get_feature(self, temporal_delay=0, normalization_type=None):
        """
        compute feature on self.data
        ****This method should generally be inherited by children classes****
        :param temporal_delay: an integer.
            if temporal_delay and (temporal_delay < self.time_delay):
                raise Exception('while using method "get_feature" temporal_delay can not be smaller than self.time_delay)
            else:
                drop the oldest (temporal_delay - self.time_delay) values in  the feature
        :param normalization_type: normalization type as explained in the method "normalize".
        :return: a numpy array of shape (Stock_num_samples - time delay, feature size)
        """
        if temporal_delay and (temporal_delay < self.time_delay):
            raise Exception('while using method "get_feature" temporal_delay can not be smaller than self.time_delay')

        if self.data is None:
            return None

        feature = self._compute_feature(self.data)

        feature = normalize(feature, normalization_type=normalization_type)

        if temporal_delay > self.time_delay:
            feature = feature[:-(temporal_delay - self.time_delay)]

        return feature

class EMA(Feature):
    """
    get EMA (Exponential moving average) of a specific column in a numpy array
    by default: column number = 3 (for close column in data array )
    """

    def __init__(self,period=10,column_num = 3, data=None, normalization_type=None, time_delay=0, is_numerical=None):
        """

        :param period: range for average (default is 10)
        :param column_num: number of column to calculate  (default is column 3 for close prices)
        :param data:
        :param normalization_type:
        :param time_delay:
        :param is_numerical:
        """
        self.data = data
        self.normalization_type = normalization_type
        self.is_numerical = 1
        self.time_delay = time_delay

        self.range = period
        self.columnNum = column_num


    def __call__(self, *args, **kwargs):
        """Don't mind it, for later use"""
        self.__init__(*args, **kwargs)

    def _compute_feature(self, data):
        column = data[:, self.columnNum]
        df = pd.DataFrame(column)
        ema_short = df.ewm(span=self.range, adjust=False).mean()
        rtrn = np.array(ema_short.values)[self.range-1:]
        return rtrn.reshape(1,rtrn.shape[0])

    def get_feature(self, temporal_delay=0, normalization_type=None):
        """
        compute feature on self.data
        ****This method should generally be inherited by children classes****
        :param temporal_delay: an integer.
            if temporal_delay and (temporal_delay < self.time_delay):
                raise Exception('while using method "get_feature" temporal_delay can not be smaller than self.time_delay)
            else:
                drop the oldest (temporal_delay - self.time_delay) values in  the feature
        :param normalization_type: normalization type as explained in the method "normalize".
        :return: a numpy array of shape (Stock_num_samples - time delay, feature size)
        """
        if temporal_delay and (temporal_delay < self.time_delay):
            raise Exception('while using method "get_feature" temporal_delay can not be smaller than self.time_delay')

        if self.data is None:
            return None

        feature = self._compute_feature(self.data)

        feature = normalize(feature, normalization_type=normalization_type)

        if temporal_delay > self.time_delay:
            feature = feature[:-(temporal_delay - self.time_delay)]

        return feature


