from ..core.dataset import Dataset
import numpy as np


class Generator(Dataset):
    _stats = None
    num_classes: int
    _val_mode: bool
    _raw_data: None
    _processed_data: None
    sample_len: int
    window_size: int
    step_size: int

    def init(self, num_classes=3, val_mode=False, sample_len=10000, window_size=10, step_size=10):
        self.num_classes = num_classes
        self._val_mode = val_mode
        self.sample_len = sample_len
        self.window_size = window_size
        self.step_size = step_size

        self._raw_data = None
        self._processed_data = None

    @property
    def raw_data(self):
        if self._raw_data is None:
            print('raw dataing')
            x_train = np.sin(range(self.sample_len))
            y_train = np.round(x_train)
            print('x_train: {}'.format(x_train))
            print('y_train: {}'.format(y_train))
            # x_test =
            # y_test =
            self._raw_data = ((x_train, y_train), (x_train, y_train))[1 * self.val_mode]
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
        return dict(num_classes=3, val_mode=False)

    @property
    def val_mode(self) -> bool:
        return self._val_mode

    # TODO fix for when the numbers dont divide well
    @property
    def input_shape(self) -> tuple:
        return self.window_size, int((self.sample_len / self.step_size) - (self.window_size / self.step_size) + 1)

    @property
    def output_shape(self):
        return (self.num_classes,)

    # TODO the 6 is arbitrary, find a better way
    def __len__(self) -> int:
        print('len: {}'.format(self.sample_len))
        return self.sample_len

    def __str__(self) -> str:
        s = 'generated_data_'
        s += ('val' if self.val_mode else 'train')
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

