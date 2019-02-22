from numpy import ndarray, float32
import numpy as np
from warnings import warn


class Data:
    def __init__(self, experiment=None, num_classes=10, normalization_type=None, **params):
        self.config = {**dict(num_classes=num_classes, normalization_type=normalization_type), **params}
        self.experiment = experiment

        self.num_classes = num_classes
        self.normalization_type = normalization_type

        self.getter_ind = 0

        self._normalization_params = {}
        self._raw_dataset = None
        self._train_size = None
        self._val_size = None

        self._train_x = None
        self._train_y = None
        self._val_x = None
        self._val_y = None

    @property
    def raw_dataset(self):
        # todo: implement for other datasets, maybe as subclasses
        if self._raw_dataset is None:
            from keras.datasets import mnist
            self._raw_dataset = mnist.load_data()
            self._raw_dataset[0][0][:] = self._raw_dataset[0][0][:]/255

        return self._raw_dataset

    def process_outputs(self, label):
        # todo: implement options
        return one_hot(label, self.num_classes)

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
        if self._train_size is None:
            self._train_size = len(self.raw_dataset[0][0])
        return self._train_size

    @property
    def val_size(self):
        if self._val_size is None:
            self._val_size = len(self.raw_dataset[1][0])
        return self._val_size

    def normalization_params(self, mode='train'):
        if mode == 'val':
            warning = 'validation normalization params should not be used to evaluate the model'
            warn(warning)

        if mode not in self._normalization_params.keys():
            def _params():
                data = self.raw_dataset[{'train': 0, 'val': 1}[mode]][0]
                return dict(mean=data.mean(), std=data.std())

            self._normalization_params[mode] = _params()
        return self._normalization_params[mode]

    def normalize(self, x, according_to='train'):
        return normalize(x, **self.normalization_params(mode=according_to))

    def denormalize(self, x, according_to='train'):
        return normalize(x, **self.normalization_params(mode=according_to))

    def set_getter_mode(self, mode='train'):
        self.getter_ind = {'train': 0, 'val': 1}[mode]

    def __getitem__(self, item):
        x, y = self.raw_dataset[self.getter_ind]
        return x[item], y[item]

    def visualize_sample(self, index):
        raise NotImplementedError()

    @property
    def callbacks(self):
        return []


def normalize(x, **normalization_params):
    # todo: add options
    x = x - normalization_params['mean']
    if normalization_params['std']:
        x /= normalization_params['std']

    return x


def denormalize(x, **normalization_params):
    # todo:
    x = x * normalization_params['std']
    x += normalization_params['mean']
    return x


def one_hot(y, num_classes):
    if isinstance(y, int):
        return np.ndarray([1 if y == i else 0 for i in range(num_classes)])

    if not isinstance(y, np.ndarray):
        y = np.ndarray(y)

    if len(y.shape) == 1:
        pass
    elif len(y.shape) == 2:
        if y.shape[1] == 1:
            y = y[:, 0]
        elif y.shape[0] == 1:
            y = y[0]
        else:
            raise RuntimeError()
    else:
        raise RuntimeError()

    return 1 * (
        np.tile(np.array([[i for i in range(num_classes)]]), (y.shape[0], 1)) == np.tile(y[:, np.newaxis], (1, num_classes))
    )

