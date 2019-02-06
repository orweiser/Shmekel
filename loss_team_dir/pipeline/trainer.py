class Trainer:
    def __init__(self, experiment, optimizer='adam', noise=None, batch_size=1024, epochs=50,
                 callbacks=None, include_experiment_callbacks=True,
                 steps_per_epoch=None, validation_steps=None, **params):
        self.config = {**dict(optimizer=optimizer, noise=noise, batch_size=batch_size, epochs=epochs,
                              callbacks=callbacks, include_experiment_callbacks=include_experiment_callbacks,
                              steps_per_epoch=steps_per_epoch, validation_steps=validation_steps),
                       **params}
        self.experiment = experiment
        self.optimizer = optimizer
        self._history = None
        self.noise = noise

        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps

        self.batch_size = batch_size
        self.epochs = epochs
        if not include_experiment_callbacks:
            self.callbacks = callbacks
        else:
            self.callbacks = experiment.get_exp_callbacks() + (callbacks or [])

        self.params = params or {}
        # todo: warn about extra params

    @property
    def history(self):
        """
        :rtype: dict
        :return:
        """
        if self._history is None:
            raise AttributeError("Trainer has history only after fit() was called")

        return self._history.history

    def noise_adder(self, gen):
        noise = self.noise
        raise NotImplementedError()

    @property
    def train_gen(self):
        raise NotImplementedError()

    def fit(self):
        exp = self.experiment
        loss = exp.loss
        metrics = exp.metrics
        data = exp.data
        model = exp.model

        batch_size = self.batch_size

        model.compile(self.optimizer, loss.loss_function, metrics)

        train_gen = self.noise_adder(data.generator(mode='train'))
        val_gen = self.noise_adder(data.generator(mode='val'))

        steps_per_epoch = self.steps_per_epoch or (data.train_size // batch_size)
        validation_steps = self.validation_steps or (data.val_size // batch_size)

        self._history = model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch,
                                            epochs=self.epochs,
                                            callbacks=self.callbacks,
                                            validation_data=val_gen, validation_steps=validation_steps, **self.params)

