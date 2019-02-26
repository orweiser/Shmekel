import numpy as np


def ind_generator(num_samples, randomize=True):
    f = np.random.permutation if randomize else range(num_samples)
    while True:
        for i in f(num_samples):
            yield i


def batch_generator(dataset, batch_size=1024, randomize=None, num_samples=None):
    # todo: add support for multiple inputs / outputs
    # todo: add augmentations support

    num_samples = num_samples or len(dataset)

    ind_gen = ind_generator(num_samples=num_samples, randomize=randomize)

    batch_x = np.ndarray((batch_size,) + dataset.input_shape)
    batch_y = np.ndarray((batch_size,) + dataset.output_shape)

    while True:
        for a in [batch_x, batch_y]:
            if a is None:
                continue
            a[:] = 0

        for i, ind in enumerate(ind_gen):
            sample = dataset[ind]

            batch_x[i] = sample['inputs']
            batch_y[i] = sample['outputs']

            if i == (batch_size - 1):
                break

        yield batch_x, batch_y