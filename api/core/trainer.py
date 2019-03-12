from api.utils.augmentations import AugmentationList
from api.utils.data_utils import batch_generator


class Trainer:
    def __init__(self, experiment=None,
                 optimizer='adam', batch_size=1024, epochs=5, augmentations=None,
                 callbacks=None, include_experiment_callbacks=True, randomize=True,
                 steps_per_epoch=None, validation_steps=None, **params):

        params['epochs'] = epochs
        self.config = {**dict(optimizer=optimizer, batch_size=batch_size, randomize=randomize,
                              augmentations=augmentations,
                              callbacks=callbacks, include_experiment_callbacks=include_experiment_callbacks,
                              steps_per_epoch=steps_per_epoch, validation_steps=validation_steps), **params}

        self.experiment = experiment
        self.optimizer = optimizer
        self._history = None
        self.randomize = randomize

        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps

        self.batch_size = batch_size
        if not include_experiment_callbacks:
            self.callbacks = callbacks
        else:
            self.callbacks = experiment.callbacks + (callbacks or [])

        self.params = params or {}
        self.user_augmentations = augmentations
        self._augmentations = None

    @property
    def augmentations(self):
        if self._augmentations is None:
            self._augmentations = AugmentationList(self.user_augmentations or [])
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

    def fit(self):
        exp = self.experiment
        loss = exp.loss
        metrics = exp.metrics
        train_dataset = exp.train_dataset
        val_dataset = exp.val_dataset
        print('val_dataset: {}'.format(val_dataset))
        model = exp.model

        batch_size = self.batch_size

        # todo: a compile method, either in Model or in Experiment
        model.compile(self.optimizer, loss, metrics, loss_weights=loss.loss_weights)

        train_gen = batch_generator(train_dataset, batch_size=batch_size, randomize=self.randomize,
                                    augmentations=self.augmentations)
        val_gen = batch_generator(val_dataset, batch_size=batch_size, randomize=self.randomize,
                                  augmentations=self.augmentations)

        steps_per_epoch = self.steps_per_epoch or (len(train_dataset) // batch_size)
        validation_steps = self.validation_steps or (len(val_dataset) // batch_size)
        print('self.validation_steps: {}'.format(self.validation_steps))
        print('len(val_dataset): {}'.format(len(val_dataset)))
        print('batch_size: {}'.format(batch_size))
        print('validation_steps: {}'.format(validation_steps))
        print('validation_steps: {}'.format(str(self.params)))

        self._history = model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch,
                                            callbacks=self.callbacks,
                                            validation_data=val_gen, validation_steps=validation_steps,
                                            **self.params)

