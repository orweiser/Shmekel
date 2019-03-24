from utils.generic_utils import one_hot
from ..core.dataset import Dataset
import numpy as np


class MNIST(Dataset):
    _stats = None
    num_classes: int
    _val_mode: bool
    _raw_data: None
    _processed_data: None

    def init(self, num_classes=10, val_mode=False):
        self.num_classes = num_classes
        self._val_mode = val_mode

        self._raw_data = None
        self._processed_data = None

    @property
    def raw_data(self):
        if self._raw_data is None:
            from keras.datasets import mnist
            self._raw_data = mnist.load_data()[1 * self.val_mode]
        return self._raw_data

    @property
    def processed_data(self):
        if self._processed_data is None:
            data = self.raw_data
            self._processed_data = self.process_inputs(data[0]), self.process_outputs(data[1])

        return self._processed_data

    def process_outputs(self, label):
        return one_hot(label, self.num_classes)

    def process_inputs(self, inputs):
        return normalize(inputs)

    def __getitem__(self, index) -> dict:
        x, y = self.processed_data

        return {
            'inputs': x[index], 'outputs': y[index]
        }

    def get_default_config(self) -> dict:
        return dict(num_classes=10, val_mode=False)

    @property
    def val_mode(self) -> bool:
        return self._val_mode

    @property
    def input_shape(self) -> tuple:
        return 28, 28

    @property
    def output_shape(self):
        return (self.num_classes,)

    def __len__(self) -> int:
        return (1000 if self.val_mode else 6000) * self.num_classes

    def __str__(self) -> str:
        s = 'mnist_'
        s += ('val' if self.val_mode else 'train')

        if self.num_classes != 10:
            s += '_with_' + str(self.num_classes) + '_classes'

        return s


def normalize(x):
    mean = np.mean(x)
    std = np.std(x)

    x = x - mean
    if std:
        x /= std

    return x


# def denormalize(x, **normalization_params):
#     # todo:
#     x = x * normalization_params['std']
#     x += normalization_params['mean']
#     return x


