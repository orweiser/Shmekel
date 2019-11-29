import numpy as np
from Utils.logger import logger


class InvalidShape(Exception):
    pass


def get_augmentation(aug):
    if isinstance(aug, BaseAugmentation):
        return aug

    if isinstance(aug, str):
        aug_str, params = aug, {}
    elif isinstance(aug, (list, tuple)):
        aug_str, params = aug
    elif isinstance(aug, dict):
        aug_str, params = [aug[key] for key in ['name', 'params']]
    else:
        raise TypeError('unexpected type of augmentation. got ' + str(type(aug)))

    return globals().get(aug_str)(**params)


class AugmentationList:
    def __init__(self, augmentations):
        if not isinstance(augmentations, (list, tuple)):
            augmentations = [augmentations]

        self.aug_list = [get_augmentation(aug) for aug in augmentations]

        for aug in self.aug_list:
            self._test_augmentation(aug)

    def get_output_shapes(self, input_shape: tuple, labels_shape: tuple):
        for aug in self.aug_list:
            input_shape, labels_shape = aug.get_output_shapes(input_shape, labels_shape)
        return input_shape, labels_shape

    def __call__(self, batch_inputs: np.ndarray, batch_labels: np.ndarray):
        for aug in self.aug_list:
            batch_inputs, batch_labels = aug(batch_inputs, batch_labels)

        return batch_inputs, batch_labels

    @logger.debug_dec
    def _test_augmentation(self, aug):
        get_shapes_test_cases = [
            ((1, 1), (11, 13)),
            ((2, 3, 4, 5), (8,)),
            ((15,), (2, 3, 4, 1, 1))
        ]

        for input_shape, output_shape in get_shapes_test_cases:
            try:
                out_shapes = aug.get_output_shapes(input_shape, output_shape)

                assert len(out_shapes) == 2, 'Augmentations "get_output_shapes" must return two outputs: ' \
                                       'new_input_shape, new_output_shape'

                for shape in out_shapes:
                    assert isinstance(shape, tuple), 'Shapes are expected to be tuples'

                for i in out_shapes[0] + out_shapes[1]:
                    if i is None: continue

                    assert isinstance(i, int), 'shape must be a tuple of integers. got ' + str(type(i))

                input = np.random.random(input_shape)
                labels = np.random.random(output_shape)

                outs = aug(input, labels)

                for o, s in zip(outs, out_shapes):
                    if s != o.shape:
                        raise ValueError('augmentations "call" outputs does not match "get_output_shapes" method')

            except InvalidShape: pass

            except NotImplementedError as e:
                logger.error('augmentations must implement "call" and "get_output_shapes"')
                raise e


""" Augmentations Base class """


class BaseAugmentation:
    def __call__(self, batch_inputs, batch_labels):
        return self.call(batch_inputs, batch_labels)

    def get_output_shapes(self, inputs_shape: tuple, labels_shape: tuple):
        raise NotImplementedError()

    def call(self, batch_inputs, batch_labels):
        raise NotImplementedError()


""" Augmentations: """


class GaussianNoise(BaseAugmentation):
    def __init__(self, mean=0, std=1, multiplicative=False):
        self.mean = mean
        self.std = std
        self.multiplicative = multiplicative

    def get_output_shapes(self, inputs_shape: tuple, labels_shape: tuple):
        return inputs_shape, labels_shape

    def get_params(self, num_samples=None):
        assert not hasattr(self.mean, '__len__'), 'need to implement support for random mean'
        assert not hasattr(self.std, '__len__'), 'need to implement support for random std'

        return self.mean, self.std

    def call(self, batch_inputs: np.ndarray, batch_labels):
        noise = np.random.normal(0, 1, size=batch_inputs.shape)

        mean, std = self.get_params(len(batch_inputs))

        noise *= std
        noise += mean

        batch_inputs += noise

        return batch_inputs, batch_labels


class ExpandLastLabelDim(BaseAugmentation):
    """ Augmentation for adding a specified amount (or 1) of zeros to the last dim of the input_labels"""
    def __init__(self, expansion=1):
        self.expansion = expansion

    def get_output_shapes(self, inputs_shape: tuple, labels_shape: tuple):
        new_shape = list(labels_shape)
        new_shape[-1] += self.expansion
        return inputs_shape, tuple(new_shape)

    def call(self, batch_inputs: np.ndarray, batch_labels: np.ndarray):
        return batch_inputs, np.concatenate([batch_labels, np.zeros(batch_labels.shape[:-1] + (self.expansion,))], axis=-1)


class ConcatExtraNoiseSamples(BaseAugmentation):

    def __init__(self, mean: float = 0, std: float = 1, fraction_to_added: float = 0.1,
                 add_extra_label_dim: bool = True, noise_class: int = -1):
        self.mean = mean
        self.std = std
        self.fraction_to_added = fraction_to_added
        self.add_extra_label_dim = add_extra_label_dim
        self.noise_class = noise_class

    def get_additional_noise_len(self, input_shape: tuple) -> int:
        return int(np.ceil(input_shape[0] * self.fraction_to_added))

    def get_output_shapes(self, inputs_shape: tuple, labels_shape: tuple) -> tuple:
        additional_noise = self.get_additional_noise_len(inputs_shape)

        inputs_shape = list(inputs_shape)
        labels_shape = list(labels_shape)

        labels_shape[-1] += self.add_extra_label_dim

        if labels_shape.__len__() == 1:
            labels_shape = [1, labels_shape[0]]

        inputs_shape[0] += additional_noise
        labels_shape[0] += additional_noise

        return tuple(inputs_shape), tuple(labels_shape)

    def call(self, batch_inputs: np.ndarray, batch_labels: np.ndarray):
        additional_noise = int(self.get_additional_noise_len(batch_inputs.shape))

        # batch_input expansion
        noise = np.random.normal(self.mean, self.std,
                                 size=(additional_noise,) + batch_inputs.shape[1:])
        batch_inputs = np.concatenate((batch_inputs, noise), axis=0)

        # batch_labels expansion
        extra_label = np.zeros((additional_noise,) + batch_labels.shape[1:-1] +
                               (batch_labels.shape[-1] + self.add_extra_label_dim,), dtype=bool)
        extra_label[..., self.noise_class] = True

        if self.add_extra_label_dim:
            extra_layer = np.zeros(batch_labels.shape[:-1] + (1,), dtype=bool)
            batch_labels = np.concatenate((batch_labels, extra_layer), axis=-1)

        if batch_labels.shape.__len__() == 1:
            batch_labels = batch_labels[:, np.newaxis].T

        batch_labels = np.concatenate((batch_labels, extra_label), axis=0)

        return batch_inputs, batch_labels
