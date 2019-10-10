from copy import deepcopy as copy
import numpy as np


class Stock:
    """
    this class represents a stock with it's features
    it holds the data and the computations
    """

    def __init__(self, stock_tckt, data, feature_axis=None, feature_list=None):
        """
        :param stock_tckt: just the name of the stock
        :param data: optional. if not given, it loads data automatically
        :param feature_list: list of features (Feature subclasses instances) to compute for the stock
        """

        self.stock_tckt = stock_tckt
        self.features = feature_list if isinstance(feature_list, (list, tuple)) else [feature_list]

        # property holders
        self._feature_axis = feature_axis if feature_axis else -1
        self._data = data
        self._feature_matrix = None
        self._numerical_feature_list = None
        self._not_numerical_feature_list = None
        self._temporal_delays = None
        self._temporal_size = None

    def set_feature_list(self, feature_list):
        self.__init__(stock_tckt=self.stock_tckt, data=self.data, feature_list=feature_list)

    def __get_data(self):
        """
        loads the data if self.data is None, else it returns self.data
        """
        return self._data

    def __set_data(self, value):
        self._data = value

    data = property(__get_data, __set_data)

    @property
    def temporal_delays(self):
        if self._temporal_delays is None:
            time_delays = [feature.time_delay for feature in self.features]

            pos_td = 0
            if any([td > 0 for td in time_delays]):
                pos_td = max([td for td in time_delays if td > 0])

            neg_td = 0
            if any([td < 0 for td in time_delays]):
                neg_td = max([-td for td in time_delays if td < 0])

            self._temporal_delays = (pos_td, neg_td)

        return self._temporal_delays

    @property
    def temporal_size(self):
        if self._temporal_size is None:
            self._temporal_size = len(self.data[1]) - sum(self.temporal_delays)
        return self._temporal_size

    def __len__(self):
        return self.temporal_size

    @property
    def feature_matrix(self):
        if self._feature_matrix is None:
            f_list = copy(self.numerical_feature_list)
            for i, f in enumerate(f_list):
                if len(f.shape) == 1:
                    f = f[np.newaxis, :]
                    if self._feature_axis:
                        f = np.swapaxes(f, 0, self._feature_axis)

                    f_list[i] = f

            self._feature_matrix = np.concatenate(f_list, axis=self._feature_axis)
        return self._feature_matrix

    def __get_features(self, numerical=True):
        f_list = [feature for feature in self.features if feature.is_numerical is numerical]
        features = None

        for ind, f in enumerate(f_list):
            feature = f.get_feature(data=self.data, normalization_type=f.normalization_type,
                                    feature_list=features)
            # TODO do we need temporal delays? talk to Or if needed
            # feature = f.get_feature(data=self.data, temporal_delay=self.temporal_delays[0],
            #                         neg_temporal_delay=self.temporal_delays[1], normalization_type=f.normalization_type,
            #                         feature_list=features)
            if features is None:
                features = [None] * len(f_list)
            features[ind] = feature

        return features

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
        :param t_start: start time index.
        :param t_end: optional: end time index. if not specified, t_end = t_start + num_time_samples
        :param num_time_samples: optional
        :return: a numpy 2-D array
        """
        if t_end is None and num_time_samples is None:
            raise Exception('while using "slice", either t_end or num_time_samples must be specified')

        if num_time_samples:
            t_end = t_start + num_time_samples

        m = np.swapaxes(self.feature_matrix, 0, self._feature_axis)
        m = m[t_start:t_end]
        m = np.swapaxes(m, 0, self._feature_axis)

        return m
