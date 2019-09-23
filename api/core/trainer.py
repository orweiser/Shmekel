from api.utils.augmentations import AugmentationList
from api.utils.data_utils import batch_generator
from Utils.logger import logger
from keras import optimizers as keras_optimizers
from keras import callbacks as keras_callbacks


""" WELCOME TO THE TRAINER!!! """


class Trainer:
    def __init__(self, experiment=None,
                 optimizer='adam', batch_size=2048, epochs=10,
                 train_augmentations=None, val_augmentations=None,
                 callbacks=None, include_experiment_callbacks=True, randomize=True,
                 steps_per_epoch=None, validation_steps=None, **params):

        params['epochs'] = epochs
        self.config = {**dict(optimizer=optimizer, batch_size=batch_size, randomize=randomize,
                              train_augmentations=train_augmentations, val_augmentations=val_augmentations,
                              callbacks=callbacks, include_experiment_callbacks=include_experiment_callbacks,
                              steps_per_epoch=steps_per_epoch, validation_steps=validation_steps), **params}

        self.experiment = experiment
        self.optimizer = parse_optimizer(optimizer)
        self._history = None
        self.randomize = randomize

        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps

        self.batch_size = batch_size
        if not include_experiment_callbacks:
            self.callbacks = callbacks
        else:
            self.callbacks = experiment.callbacks + (callbacks or [])

        self.callbacks = [parse_callback(c) for c in self.callbacks]

        self.params = params or {}
        self.user_augmentations = {'train': train_augmentations, 'val': val_augmentations}
        # self._augmentations = {'train': None, 'val': None}
        self._augmentations = None

    @property
    def augmentations(self):
        if self._augmentations is None:
            self._augmentations = {}
            for key in ('train', 'val'):
                self._augmentations[key] = AugmentationList(self.user_augmentations[key] or [])
        return self._augmentations

    @property
    def history(self):
        """
        :rtype: dict
        :return:
        """
        if self._history is None:
            raise AttributeError("Trainer has history only after fit() was called")

        return self._history.history

    @logger.info_dec
    def fit(self):
        exp = self.experiment
        loss = exp.loss
        metrics = exp.metrics
        train_dataset = exp.train_dataset
        val_dataset = exp.val_dataset
        model = exp.model

        batch_size = self.batch_size

        # todo: a compile method, either in Model or in Experiment
        model.compile(self.optimizer, loss, metrics, loss_weights=loss.loss_weights)

        train_gen = batch_generator(train_dataset, batch_size=batch_size, randomize=self.randomize,
                                    augmentations=self.augmentations['train'])
        val_gen = batch_generator(val_dataset, batch_size=batch_size, randomize=self.randomize,
                                  augmentations=self.augmentations['val'])

        steps_per_epoch = self.steps_per_epoch or (len(train_dataset) // batch_size)
        validation_steps = self.validation_steps or (len(val_dataset) // batch_size)

        logger.info('Enter fitting loop')
        self._history = model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch,
                                            callbacks=self.callbacks,
                                            validation_data=val_gen, validation_steps=validation_steps,
                                            **self.params)
        logger.info('Exit fitting loop')


def _parse_item(item):
    if isinstance(item, str):
        return item, {}
    if isinstance(item, dict):
        return item['name'], {key: val for key, val in item if key != 'name'}
    if isinstance(item, (list, tuple)):
        assert len(item) == 2
        assert isinstance(item[0], str)
        assert isinstance(item[1], dict)
        return tuple(item)


def parse_optimizer(item):
    if isinstance(item, keras_optimizers.Optimizer):
        return item

    opt_name, opt_params = _parse_item(item)

    if (not opt_params) and opt_name in ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam']:
        # using keras aliases and defaults
        return opt_name

    module = getattr(keras_optimizers, opt_name)
    return module(**opt_params)


def parse_callback(item):
    if isinstance(item, keras_callbacks.Callback):
        return item

    c_name, c_params = _parse_item(item)

    module = getattr(keras_callbacks, c_name)
    return module(**c_params)

