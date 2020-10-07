from shmekel_core import Feature
import numpy as np


class Candle(Feature):
    def __init__(self, with_volume=True, time_delay=0, **kwargs):
        super(Candle, self).__init__(**kwargs)

        self.is_numerical = True
        self.time_delay = time_delay
        self.with_volume = with_volume
        self.num_features = 5 if with_volume else 4

    def _compute_feature(self, data, feature_list=None):
        if self.with_volume:
            return data[0][self.time_delay:]

        a = np.swapaxes(data[0], 0, self._feature_axis)
        ind = [i for i in range(len(self._pattern)) if i != self._pattern.index('volume')]
        a = a[ind]

        a = np.swapaxes(a, 0, self._feature_axis)
        return a[self.time_delay:]


class RawCandle(Candle):
    def __init__(self, *args, **kwargs):
        super(RawCandle, self).__init__(*args, **kwargs)

        self.is_numerical = False
        self.normalization_type = None
        self.with_volume = True
        self.num_features = 5


class AltCandle(Feature):
    def __init__(self, with_volume=True, time_delay=0, **kwargs):
        super(AltCandle, self).__init__(**kwargs)

        self.is_numerical = True
        self.time_delay = time_delay
        self.with_volume = with_volume
        self.num_features = 5 if with_volume else 4

    def _compute_feature(self, data, feature_list=None, ):
        if self.with_volume:
            for i in [0, 1, 3]:
                data[0][:, i] = np.divide(data[0][:, i], data[0][:, 2])
            return data[0][self.time_delay:]

        a = np.swapaxes(data[0], 0, self._feature_axis)
        ind = [i for i in range(len(self._pattern)) if i != self._pattern.index('volume')]
        a = a[ind]

        a = np.swapaxes(a, 0, self._feature_axis)
        for i in [0, 1, 3]:
            a[0][:, i] = np.divide(a[0][:, i], a[0][:, 2])
        return a[self.time_delay:]


class DateTuple(Feature):
    def __init__(self, normalization_type=None, **kwargs):
        super(DateTuple, self).__init__(normalization_type=normalization_type, **kwargs)

        self.time_delay = 0
        self.is_numerical = False
        self.num_features = 1

    def _compute_feature(self, data, feature_list=None):
        return data[1]


class High(Feature):
    def __init__(self, normalization_type='default', **kwargs):
        super(High, self).__init__(normalization_type=normalization_type, **kwargs)

        self.time_delay = 0
        self.is_numerical = True
        self.num_features = 1

    def _compute_feature(self, data, feature_list=None):
        return self._get_basic_feature(data[0], 'High')


class Low(Feature):
    def __init__(self, normalization_type='default', **kwargs):
        super(Low, self).__init__(normalization_type=normalization_type, **kwargs)

        self.time_delay = 0
        self.is_numerical = True
        self.num_features = 1

    def _compute_feature(self, data, feature_list=None):
        return self._get_basic_feature(data[0], 'Low')


class Open(Feature):
    def __init__(self, normalization_type='default', **kwargs):
        super(Open, self).__init__(normalization_type=normalization_type, **kwargs)

        self.time_delay = 0
        self.is_numerical = True
        self.num_features = 1

    def _compute_feature(self, data, feature_list=None):
        return self._get_basic_feature(data[0], 'Open')


class Close(Feature):
    def __init__(self, normalization_type='default', **kwargs):
        super(Close, self).__init__(normalization_type=normalization_type, **kwargs)

        self.time_delay = 0
        self.is_numerical = True
        self.num_features = 1

    def _compute_feature(self, data, feature_list=None):
        return self._get_basic_feature(data[0], 'Close')


class Volume(Feature):
    def __init__(self, normalization_type='default', **kwargs):
        super(Volume, self).__init__(normalization_type=normalization_type, **kwargs)

        self.time_delay = 0
        self.is_numerical = True
        self.num_features = 1

    def _compute_feature(self, data, feature_list=None):
        return self._get_basic_feature(data[0], 'Volume')
