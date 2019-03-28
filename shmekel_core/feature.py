import numpy as np

#
# feature_axis = get_config()['feature_axis']
# pattern = get_config()['pattern']

# default_pattern = ('Open', 'High', 'Low', 'Close', 'Volume')
# default_feature_axis = -1


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
    def __init__(self, pattern=('open', 'high', 'low', 'close', 'volume'), feature_axis=-1, normalization_type=None,
                 time_delay=0, num_features=1, is_numerical=True):
        """
        :param pattern: the pattern of the candle data inserted ('High', 'Low'...), as defined in the config file
        :type pattern: list
        :param feature_axis: axis orientation of the data, as defined in the config file
        :type feature_axis: int
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
        self._pattern = pattern
        self._feature_axis = feature_axis
        self.normalization_type = normalization_type
        self.is_numerical = is_numerical
        self.time_delay = time_delay
        self.num_features = num_features

    def _get_basic_feature(self, candles, key, keep_dims=False):
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

        candles = np.swapaxes(candles, 0, self._feature_axis)
        f = candles[self._pattern.index(key.lower())]

        if keep_dims:
            f = np.swapaxes(f[np.newaxis, :], 0, self._feature_axis)

        return f

    def _compute_feature(self, data):
        """
        This is the core function of the feature. use it to define the feature's functionality
        NOTE: do not edit the code here, instead use inheritance to create sub classes with different
            "_compute_feature" definitions.

        ****This method must be overridden by children classes****
        :rtype: ndarray
        """
        raise NotImplementedError()

    def get_feature(self, data, temporal_delay=0, neg_temporal_delay=0, normalization_type=None):
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

        if self.is_numerical and len(feature.shape) > 1 and self._feature_axis:
            feature = np.swapaxes(feature, -1, self._feature_axis)

        if self.time_delay > 0:
            temporal_delay = temporal_delay - self.time_delay * (temporal_delay >= self.time_delay)

        if temporal_delay:
            feature = feature[temporal_delay:]

        if self.time_delay < 0:
            neg_temporal_delay = neg_temporal_delay - abs(self.time_delay) * (neg_temporal_delay >= abs(self.time_delay))

        if neg_temporal_delay:
            feature = feature[neg_temporal_delay:]

        if self.is_numerical and len(feature.shape) > 1 and self._feature_axis:
            feature = np.swapaxes(feature, -1, self._feature_axis)

        return feature
