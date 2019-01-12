import numpy as np
from numpy import ndarray
from data import load_stock
from shmekel_config import get_config
from copy import deepcopy as copy


"""
Hi, in this file we define the general classes of Features and Stocks.
for now, please ignore the StockList class.

in implementing the features, your main focus will be on defining subclasses of Feature. 
the real must do is to redefine the "_compute_feature" method.
go to the end of the file for pseudo-example.

Stock.load_data must be implemented. Rotem said it might be easier to directly download data each time.
    check it out and decide on the best way to load data.
    note that there need to be some notion of train and validation sets

we need to decide on normalization methods. if you get to it, do implement it
but if not it's ok because it should not be feature specific
    
last thing, note that in Stock.build i try to define self.temporal_size, once you do load the data in some pattern,
    make sure to finish the definition there
    
"""


feature_axis = get_config()['feature_axis']


def normalize(feature, normalization_type=None):
    """
    different normalization methods.

    :param feature: a numpy array to normalize
    :param normalization_type: some identification of the normalization type
    :return: a normalized numpy array of the same shape

    note: though normalization is important, we did not decide on types exactly. therefore, this
    method is not necessary at first stage.
    """

    normalization_type = normalization_type or normalization_type
    # todo: add normalization methods

    return feature


class Feature:
    """
    this is the general class of features
    !!! it is NOT a specific feature. specific features will be subclasses of this class !!!

    The code might require some corrections and adaptations, but in general it needs not to be changed

    """
    def __init__(self, normalization_type=None, time_delay=0, num_features=1, is_numerical=True):
        """
        :param data: None or a pointer to the stock data

        :param normalization_type:

        :param time_delay: an integer that specifies the number of past samples needed to compute
                    the an entry .note that if N samples are required, then the output feature
                    vector is N samples shorter than the original data

        :param is_numerical: a boolean value. True if the feature is numerical and False otherwise.
                    for example, "date" is a non-numerical feature because the usual adding and
                    multiplication does not make sense.
                    However, we might want a different variation on "Date" that is numerical, for
                    example we could define a feature to indicate the day of the week via 1-hot
                    representation - more on that in another time
        """
        self.normalization_type = normalization_type
        self.is_numerical = is_numerical
        self.time_delay = time_delay
        self.num_features = num_features

    def _compute_feature(self, data):
        """
        This is the core function of the feature. use it to define the feature's functionality
        NOTE: do not edit the code here, instead use inheritance to create sub classes with different
            "_compute_feature" definitions.

        ****This method must be overridden by children classes****

        :param data:
        :return:
        :rtype: ndarray
        """
        pass

    def get_feature(self, data, temporal_delay=0, normalization_type=None):
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

        feature = self._compute_feature(data)
        feature = normalize(feature, normalization_type=normalization_type)

        if len(feature.shape) > 1 and feature_axis:
            feature = np.swapaxes(feature, 0, feature_axis)

        if temporal_delay > self.time_delay:
            feature = feature[:-(temporal_delay - self.time_delay)]

        if len(feature.shape) > 1 and feature_axis:
            feature = np.swapaxes(feature, 0, feature_axis)

        return feature


class Stock:
    """
    this class represents a stock with it's features, holds the data and the computations
    you need to de two things here:
        1. implement the method "load_data"
        2. define self.temporal_size in method "build"
    """
    def __init__(self, stock_tckt, data=None, feature_list=None):
        """
        :param stock_tckt: just the name of the stock or whatever identifier you decide is best
        :param data: optional if "load_data" is implemented.
        :param feature_list: list of features (Feature subclasses) to compute for the stock
        :param normalization_types: the normalization types to use on each feature.
            if normalization_types is a list, it should be the same length as feature_list.
            else, it is used on all features
        :param validation: boolean. if true, method "load_data" will load validation data instead of train data
        """
        self.stock_tckt = stock_tckt
        self.features = feature_list if type(feature_list) is list else [feature_list]

        # property holders
        self._data = data
        self._feature_matrix = None
        self._numerical_feature_list = None
        self._not_numerical_feature_list = None
        self._temporal_delay = None
        self._temporal_size = None

    def reset_properties(self, reset_data=True):
        if reset_data:
            self.__init__(stock_tckt=self.stock_tckt, feature_list=self.features)
        else:
            self.__init__(stock_tckt=self.stock_tckt, feature_list=self.features, data=self._data)

    def load_data(self, override=False):
        """
        loads the data if self.data is None, else it returns self.data
        """
        if self._data is None or override:
            stock = self.stock_tckt
            self._data = load_stock(stock)
        return self._data

    def __set_data(self, value):
        self._data = value

    data = property(load_data, __set_data)

    @property
    def temporal_delay(self):
        if self._temporal_delay is None:
            self._temporal_delay = max([feature.time_delay for feature in self.features])
        return self._temporal_delay

    @property
    def temporal_size(self):
        if self._temporal_size is None:
            self._temporal_size = len(self.data[1]) - self.temporal_delay
        return self._temporal_size

    @property
    def feature_matrix(self):
        if self._feature_matrix is None:
            f_list = copy(self.numerical_feature_list)
            for i, f in enumerate(f_list):
                if len(f.shape) == 1:
                    f = f[np.newaxis, :]
                    if feature_axis:
                        f = np.swapaxes(f, 0, feature_axis)

                    f_list[i] = f

            self._feature_matrix = np.concatenate(f_list, axis=feature_axis)
        return self._feature_matrix

    def __get_features(self, numerical=True):
        f_list = [feature for feature in self.features if feature.is_numerical is numerical]
        return [f.get_feature(data=self.data, temporal_delay=self.temporal_delay) for f in f_list]

    @property
    def numerical_feature_list(self):
        if not self._numerical_feature_list:
            self._numerical_feature_list = self.__get_features(numerical=True)
        return self._numerical_feature_list

    @property
    def not_numerical_feature_list(self):
        if not self._not_numerical_feature_list:
            self._not_numerical_feature_list = self.__get_features(numerical=False)
        return self._not_numerical_feature_list

    def slice(self, t_start, t_end=None, num_time_samples=None):
        """
        computes a slice of the full feature matrix
        :param t_start: start time index
        :param t_end: optional: end time index. if not specified, t_end = t_start + num_time_samples
        :param num_time_samples: optional
        :return: a numpy 2-D array
        """
        if t_end is None and num_time_samples is None:
            raise Exception('while using "slice", either t_end or num_time_samples must be specified')

        if num_time_samples:
            t_end = t_start + num_time_samples

        return self.feature_matrix[t_start:t_end]


# class StockList:
#     """
#     NOTE: this class is not yet thought of. ignore it for now.
#     """
#     def __init__(self, stock_tckt_list, features_list, normalization_types=None):
#         def L(x):
#             if type(x) is list:
#                 return copy(x)
#             else:
#                 return [copy(x) for _ in range(num_features)]
#
#         num_features = 1 if type(features_list) is not list else len(features_list)
#
#         self.stock_tckt_list = L(stock_tckt_list)
#         self.normalization_types = L(normalization_types)
#         self.feature_list = L(features_list)
#
#         for f in self.feature_list:
#             if not issubclass(f, Feature):
#                 raise Exception('all features in feature_list must be a subclass of Feature')
#
#         self.built = False
#         self.stock_list = None
#
#     def build(self):
#         if self.built:
#             return
#
#         self.stock_list = []
#         for tckt in self.stock_tckt_list:
#             self.stock_list.append(
#                 Stock(stock_tckt=tckt, feature_list=self.feature_list, normalization_types=self.normalization_types)
#             )
#
#         self.built = True
#
#     def generator(self, batch_size=512, time_length=32, randomize=True):
#         pass


"""Down here we define the features"""


class __feature_example__(Feature):
    """
    this is an example on how to create a feature subclass.
    use this green space to describe the feature for those of us who are not familiar
    """
    def __init__(self, data=None, normalization_type=None):  # DO NOT CHANGE THE DECLARATION
        """
        use this method to define the parameters of the feature
        """

        self.time_delay = 0  # change it according to the feature as described in class Feature
        self.is_numerical = None  # change it according to the feature as described in class Feature

        # here you can define more parameters that "_compute_feature" might need to use
        self.param1 = None
        self.param2 = None
        # ...
        self.paramN = None

        # the following line must be included
        super(__feature_example__, self).__init__(data=data, normalization_type=normalization_type,
                                                  time_delay=self.time_delay, is_numerical=self.is_numerical)

    def _compute_feature(self, data):
        """
        That's the core method of the feature. It MUST be re-defined for every feature

        define the function to output the feature as numpy array from the data

        :param data:
        :return: a numpy array
        """

        # if you defined some extra parameters, you can access them as follows:
        param1 = self.param1    # and so on...

        f = data ** 2  # just some calculations to create the feature

        return f
