import numpy as np
from numpy import ndarray
from data import load_stock
from shmekel_config import get_config
from copy import deepcopy as copy


feature_axis = get_config()['feature_axis']
pattern = get_config()['pattern']


def _get_basic_feature(candles, key, keep_dims=False):
    """
    pulls out a basic feature from candle array.
    the candle array is a 2-D with basic features as High or Open
    the pattern of features is decided in shmekel_config, as is the feature_axis

    the main idea here, is that given a candle, use this function to pull out a specific field.

    :type candles: ndarray | tuple
    :param candles: a 2-D numpy array

    :param key: one of Open, Close, High, Low, Volume
     :type key: str
    :return:
    """

    if type(candles) is tuple:
        candles = candles[0]

    candles = np.swapaxes(candles, 0, feature_axis)
    f = candles[pattern.index(key)]

    if keep_dims:
        f = np.swapaxes(f[np.newaxis, :], 0, feature_axis)

    return f


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
        this is a basic feature class and you need not to touch it
        however, do take the time and effort to understand its structure and parameters
        to see how to implement a specific feature, go to the feature example at the end
    """
    def __init__(self, normalization_type=None, time_delay=0, num_features=1, is_numerical=True):
        """
        :param normalization_type:

        :type time_delay: int
        :param time_delay: the number of past samples needed to compute
                    an entry. note that if (N + 1) samples are required, than the output feature
                    vector is N samples shorter than the original data

        :type num_features: int
        :param num_features:

        :type is_numerical: bool
        :param is_numerical: True if the feature is numerical and False otherwise.
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
        :rtype: ndarray
        """
        pass

    def get_feature(self, data, temporal_delay=0, normalization_type=None):
        """
        compute feature on self.data

        This method should be inherited by children classes, do ***not*** override!

        :param temporal_delay: an integer.
            if temporal_delay and (temporal_delay < self.time_delay):
                raise Exception('while using method "get_feature" temporal_delay can not be smaller than self.time_delay)
            else:
                drop the oldest (temporal_delay - self.time_delay) values in  the feature

        :param normalization_type: normalization type as explained in the method "normalize".
        :return: a numpy array of shape:
                (Stock_num_samples - time_delay, feature size) if feature_axis is 0
                (feature_size, Stock_num_samples - time_delay) if feature_axis is -1
        """
        if temporal_delay and (temporal_delay < self.time_delay):
            raise Exception('while using method "get_feature" temporal_delay can not be smaller than self.time_delay')

        feature = self._compute_feature(data)
        feature = normalize(feature, normalization_type=normalization_type)

        if self.is_numerical and len(feature.shape) > 1 and feature_axis:
            feature = np.swapaxes(feature, 0, feature_axis)

        if temporal_delay > self.time_delay:
            feature = feature[:-(temporal_delay - self.time_delay)]

        if self.is_numerical and len(feature.shape) > 1 and feature_axis:
            feature = np.swapaxes(feature, 0, feature_axis)

        return feature


class Stock:
    """
    this class represents a stock with it's features
    it holds the data and the computations
    """
    def __init__(self, stock_tckt, data=None, feature_list=None):
        """
        :param stock_tckt: just the name of the stock
        :param data: optional. if not given, it loads data automatically
        :param feature_list: list of features (Feature subclasses instances) to compute for the stock
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

    def set_feature_list(self, feature_list):
        self.__init__(stock_tckt=self.stock_tckt, data=self.data, feature_list=feature_list)

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
    def non_numerical_feature_list(self):
        if not self._not_numerical_feature_list:
            self._not_numerical_feature_list = self.__get_features(numerical=False)
        return self._not_numerical_feature_list

    def slice(self, t_start, t_end=None, num_time_samples=None):
        """
        computes a slice of the full feature matrix
        :param t_start: start time index.
        :param t_end: optional: end time index. if not specified, t_end = t_start + num_time_samples
        :param num_time_samples: optional
        :return: a numpy 2-D array
        """
        if t_end is None and num_time_samples is None:
            raise Exception('while using "slice", either t_end or num_time_samples must be specified')

        if num_time_samples:
            t_end = t_start + num_time_samples

        m = np.swapaxes(self.feature_matrix, 0, feature_axis)
        m = m[t_start:t_end]
        m = np.swapaxes(m, 0, feature_axis)

        return m


class __feature_example__(Feature):
    """
    this is an example on how to create a feature subclass.
    use this green space to describe the feature for those of us who are not familiar
    """
    def __init__(self, arg1=3, arg2=2, arg3=None, **kwargs):
        # be sure to start this method with the "super" call because it has some defaults that you might
        # want to override later
        super(__feature_example__, self).__init__(**kwargs)

        # once we defined the defaults through the "super" call, we can override those we want to override:
        self.is_numerical = True  # change it according to the feature as described in class Feature
        self.time_delay = 0  # change it according to the feature as described in class Feature
        self.num_features = 2  # change it according to the feature as described in class Feature

        # down here you can define more parameters that "_compute_feature" might need to use

        # here we define params that are given as arguments
        self.arg1 = arg1
        self.arg2 = arg2
        # ...
        self.arg3 = arg3

        # and here we define params that are *not* given as arguments
        self.param1 = 1
        self.param2 = 15
        # ...
        self.paramN = 42

    def _compute_feature(self, data):
        """
        That's the core method of the feature. It MUST be re-defined for every feature
        define the function to output the feature as numpy array from the data

        this an example feature that computes powers of 'High' and 'Close'
        according to arguments specified by the user
            it computes 'Close' to the power of self.arg2 and 'High' to the self.arg1 power

        :param data: a tuple that holds a numpy array to represent a candle and a list of dates.
            the numpy array is 2-D, with one axis for time and another
            for the basic features (Open, High, Low, Close, Volume)
                                    *** note that we might decide to change this pattern,
                                        so make sure not to use it explicitly, but only via the
                                        _get_basic_feature() function as seen below

            the feature axis is defined in the shmekel_config file

            if feature_axis is 0, than the shape of the array is (5, time)
            if it is -1, than (time, 5).
            the _get_basic_feature() function covers that part as well
        :type data: tuple

        :return: a numpy array
            the array should be one of the following:
                if self.num_features == 1:
                    either a 1-D array of length Time,
                    or:
                        a 2-D array, where one axis has size Time and the other size 1.
                        if feature_axis is 0, the shape is (1, Time)
                        if feature_axis is -1, the shape is (Time, 1)

                else:   (self.num_features > 1)
                    if you have num_features > 1, it means you compute more than one feature,
                    therefore your array should be 2-D

                    if feature_axis is 0, the shape is (self.num_features, Time)
                    if feature_axis is -1, the shape is (Time, self.num_features)
        :rtype: ndarray
        """

        # if you defined some extra parameters, you can access them as follows:
        high_power = self.arg1
        close_power = self.arg2

        candle, dates = data  # that is the shape of the data we get

        # here were pulling the relevant fields out of "candle"
        # without implicitly decide on a pattern for candles or the value of feature_axis
        high = _get_basic_feature(candle, 'High')
        close = _get_basic_feature(candle, 'Close')

        # computation of the features and stacking over the feature_axis axis.
        output_array = np.stack([high ** high_power, close ** close_power], axis=feature_axis)
        return output_array
