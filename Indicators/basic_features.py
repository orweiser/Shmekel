from .Indicators import Feature, _get_basic_feature, feature_axis, pattern
import numpy as np


class Candle(Feature):
    def __init__(self, with_volume=True, **kwargs):
        super(Candle, self).__init__(**kwargs)

        self.is_numerical = True
        self.time_delay = 0
        self.with_volume = with_volume
        self.num_features = 5 if with_volume else 4

    def _compute_feature(self, data):
        if self.with_volume:
            return data[0]

        a = np.swapaxes(data[0], 0, feature_axis)
        ind = [i for i in range(len(pattern)) if i != pattern.index('Volume')]
        a = a[ind]

        a = np.swapaxes(a, 0, feature_axis)
        return a


class DateTuple(Feature):
    def __init__(self, **kwargs):
        super(DateTuple, self).__init__(**kwargs)

        self.time_delay = 0
        self.is_numerical = False
        self.num_features = 1

    def _compute_feature(self, data):
        return data[1]


class High(Feature):
    def __init__(self, **kwargs):
        super(High, self).__init__(**kwargs)

        self.time_delay = 0
        self.is_numerical = True
        self.num_features = 1

    def _compute_feature(self, data):
        return _get_basic_feature(data[0], 'High')


class Low(Feature):
    def __init__(self, **kwargs):
        super(Low, self).__init__(**kwargs)

        self.time_delay = 0
        self.is_numerical = True
        self.num_features = 1

    def _compute_feature(self, data):
        return _get_basic_feature(data[0], 'Low')


class Open(Feature):
    def __init__(self, **kwargs):
        super(Open, self).__init__(**kwargs)

        self.time_delay = 0
        self.is_numerical = True
        self.num_features = 1

    def _compute_feature(self, data):
        return _get_basic_feature(data[0], 'Open')


class Close(Feature):
    def __init__(self, **kwargs):
        super(Close, self).__init__(**kwargs)

        self.time_delay = 0
        self.is_numerical = True
        self.num_features = 1

    def _compute_feature(self, data):
        return _get_basic_feature(data[0], 'Close')


class Volume(Feature):
    def __init__(self, **kwargs):
        super(Volume, self).__init__(**kwargs)

        self.time_delay = 0
        self.is_numerical = True
        self.num_features = 1

    def _compute_feature(self, data):
        return _get_basic_feature(data[0], 'Volume')