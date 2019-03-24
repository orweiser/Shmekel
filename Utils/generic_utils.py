import numpy as np


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
