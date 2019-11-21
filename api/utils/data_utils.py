import numpy as np
from copy import deepcopy as copy


def ind_generator(num_samples, randomize=True):
    f = np.random.permutation if randomize else range
    while True:
        for i in f(num_samples):
            yield i


def batch_generator(dataset, batch_size=1024, randomize=True, max_ind=None, augmentations=None,
                    ind_gen=None):
    # todo: add support for multiple inputs / outputs

    max_ind = max_ind or len(dataset)

    ind_gen = ind_gen or ind_generator(num_samples=max_ind, randomize=randomize)

    batch_x = np.ndarray((batch_size,) + dataset.input_shape)
    batch_y = np.ndarray((batch_size,) + dataset.output_shape)

    j = 0
    try:
        while True:
            for a in [batch_x, batch_y]:
                if a is None:
                    continue
                a[:] = 0

            for i, ind in enumerate(ind_gen):
                j += 1
                sample = dataset[ind]

                batch_x[i] = sample['inputs']
                batch_y[i] = sample['outputs']

                if i == (batch_size - 1):
                    break

            if augmentations:
                batch_x_o, batch_y_o = augmentations(batch_x, batch_y)
            else:
                batch_x_o, batch_y_o = [batch_x, batch_y]

            yield copy(batch_x_o), copy(batch_y_o)
    except StopIteration:
        raise StopIteration('Got StopIteration after %d samples was generated. For Parsing:%d' % (j, j))
