from numpy import ndarray, float32
import numpy as np
from warnings import warn


class Data:
    def __init__(self, experiment=None, num_classes=10, normalization_type=None, **params):
        self.config = {**dict(num_classes=num_classes, normalization_type=normalization_type), **params}
        self.experiment = experiment

        self.num_classes = num_classes
        self.normalization_type = normalization_type

        self._normalization_params = {}

    @property
    def raw_dataset(self):
        return (ndarray, ndarray), (ndarray, ndarray)

    def process_outputs(self, label):
        return label

    @property
    def train_x(self):
        """
        :rtype: ndarray
        """
        if self._train_x is None:
            self._train_x = self.normalize(self.raw_dataset[0][0])
        return self._train_x

    @property
    def train_y(self):
        """
        :rtype: ndarray
        """
        if self._train_y is None:
            self._train_y = self.process_outputs(self.raw_dataset[0][1])
        return self._train_y

    @property
    def val_x(self):
        """
        :rtype: ndarray
        """
        if self._val_x is None:
            self._val_x = self.normalize(self.raw_dataset[1][0])
        return self._val_x

    @property
    def val_y(self):
        """
        :rtype: ndarray
        """
        if self._val_y is None:
            self._val_y = self.process_outputs(self.raw_dataset[1][1])
        return self._val_y

    @property
    def train_size(self):
        return int

    @property
    def val_size(self):
        return int

    def normalization_params(self, mode='train'):
        if mode == 'val':
            warning = 'validation normalization params should not be used to evaluate the model'
            warn(warning)

        if mode not in self._normalization_params.keys():
            def _params():
                return dict(mean=float32, std=float32)

            self._normalization_params[mode] = _params()
        return self._normalization_params[mode]

    @staticmethod
    def ind_generator(num_samples, randomize=True):
        f = np.random.permutation if randomize else range
        while True:
            for i in f(num_samples):
                yield i

    def batch_generator(self, batch_size=1024, with_labels=True, use_raw_data=False, mode='train'):
        num_samples = self.train

        ind_gen = self.ind_generator()
        yield self.train_x, self.train_y

    def normalize(self, x, mode='train'):
        return normalize(x, **self.normalization_params(mode=mode))

    def denormalize(self, x, mode='train'):
        return normalize(x, **self.normalization_params(mode=mode))


def normalize(x, **normalization_params):
    return ndarray


def denormalize(x, **normalization_params):
    return ndarray


def one_hot(y, num_classes):
    return ndarray
