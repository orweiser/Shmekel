import numpy as np


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

    def get_output_shapes(self, input_shape: tuple, labels_shape: tuple):
        for aug in self.aug_list:
            input_shape, labels_shape = aug.get_output_shapes(input_shape, labels_shape)
        return input_shape, labels_shape

    def __call__(self, batch_inputs: np.ndarray, batch_labels: np.ndarray):
        for aug in self.aug_list:
            batch_inputs, batch_labels = aug(batch_inputs, batch_labels)

        return batch_inputs, batch_labels


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


class ExpandLastDim(BaseAugmentation):

    def __init__(self, expansion=1):
        self.expansion = expansion

    def get_output_shapes(self, inputs_shape: tuple, labels_shape: tuple):
        new_shape = list(labels_shape)
        new_shape[-1] += self.expansion
        return inputs_shape, tuple(new_shape)

    def call(self, batch_inputs: np.ndarray, batch_labels: np.ndarray):
        return np.concatenate([batch_labels, np.zeros((batch_labels.size // batch_labels.shape[-1], self.expansion))], axis=-1)


class AddNoise(BaseAugmentation):
    def __init__(self, mean: float = 0, std: float = 1, fraction_to_added: float = 0.1,
                 add_extra_label_dim: bool = True):
        self.mean = mean
        self.std = std
        self.fraction_to_added = fraction_to_added
        self.add_extra_label_dim = add_extra_label_dim

    def get_additional_noise(self, input_shape: tuple) -> np.ndarray:
        return np.ceil(input_shape[0] * self.fraction_to_added)

    def get_output_shapes(self, inputs_shape: tuple, labels_shape: tuple):
        additional_noise = self.get_additional_noise(inputs_shape)
        inputs_shape = inputs_shape[0] + additional_noise, + inputs_shape[1:]
        labels_shape = (labels_shape[0] + 1,) if self.add_extra_label_dim else labels_shape
        return inputs_shape, labels_shape

    def call(self, batch_inputs: np.ndarray, batch_labels: np.ndarray):
        additional_noise = self.get_additional_noise(batch_inputs.shape)
        noise = np.random.normal(self.mean, self.std,
                                 size=(additional_noise,) + batch_inputs.shape[:1])

        if self.add_extra_label_dim:
            extra_label = np.zeros((additional_noise,) + batch_labels.shape[1:])
            extra_label[:, -1] = True
        else:
            extra_label = np.zeros((additional_noise,) + batch_labels.shape[1:-1] +
                                   batch_labels.shape[-1])
            extra_label[:, -1] = True

        batch_inputs = np.concatenate((batch_inputs, noise), axis=0)
        batch_labels = np.concatenate((batch_labels, extra_label), axis=0)
        return batch_inputs, batch_labels
