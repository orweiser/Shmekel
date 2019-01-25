from numpy import ndarray, float32


class Data:
    def __init__(self, experiment=None, **params):
        self.experiment = experiment
        self._train_normalization_params = None
        self._val_normalization_params = None

    @property
    def dataset(self):
        return (ndarray, ndarray), (ndarray, ndarray)

    @property
    def train_x(self):
        """
        :rtype: ndarray
        """
        if self._train_x is None:
            self._train_x = normalize(self.dataset[0][0], self.normalization_params('train'))
        return self._train_x

    @property
    def train_y(self):
        """
        :rtype: ndarray
        """
        if self._train_y is None:
            self._train_y = one_hot(self.dataset[0][1])
        return self._train_y

    @property
    def val_x(self):
        """
        :rtype: ndarray
        """
        if self._val_x is None:
            self._val_x = normalize(self.dataset[0][0], self.normalization_params('train'))
        return self._val_x

    @property
    def val_y(self):
        """
        :rtype: ndarray
        """
        if self._val_y is None:
            self._val_y = one_hot(self.dataset[1][1])
        return self._val_y

    @property
    def train_size(self):
        return int

    @property
    def val_size(self):
        return int

    def normalization_params(self, mode='train'):
        attr_name = '_' + mode + '_normalization_params'

        if getattr(self, attr_name) is not None:
            return getattr(self, attr_name)

        def _params():
            return dict(mean=float32, std=float32)

        setattr(self, attr_name, _params())

    def generator(self, mode='train'):
        yield self.train_x, self.train_y


def normalize(x, normalization_params):
    return ndarray


def denormalize(x):
    return ndarray


def one_hot(y):
    return ndarray
