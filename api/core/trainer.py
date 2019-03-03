from api.utils.data_utils import batch_generator


class Trainer:
    def __init__(self, experiment=None,
                 optimizer='adam', batch_size=1024,
                 callbacks=None, include_experiment_callbacks=True, randomize=True,
                 steps_per_epoch=None, validation_steps=None, **params):

        if 'epochs' not in params:
            params['epochs'] = 5
        self.config = {**dict(optimizer=optimizer, batch_size=batch_size, randomize=randomize,
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
        # todo: warn about extra params
        # todo: add augmentations

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
        model = exp.model

        batch_size = self.batch_size

        # todo: a compile method, either in Model or in Experiment
        model.compile(self.optimizer, loss, metrics, loss_weights=loss.loss_weights)

        train_gen = batch_generator(train_dataset, batch_size=batch_size, randomize=self.randomize)
        val_gen = batch_generator(val_dataset, batch_size=batch_size, randomize=self.randomize)

        steps_per_epoch = self.steps_per_epoch or (len(train_dataset) // batch_size)
        validation_steps = self.validation_steps or (len(val_dataset) // batch_size)

        self._history = model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch,
                                            callbacks=self.callbacks,
                                            validation_data=val_gen, validation_steps=validation_steps,
                                            **self.params)

