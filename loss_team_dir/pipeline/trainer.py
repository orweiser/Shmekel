import numpy as np


class Trainer:
    def __init__(self, experiment, optimizer='adam', noise=None, batch_size=1024,
                 callbacks=None, include_experiment_callbacks=True, randomize=True,
                 steps_per_epoch=None, validation_steps=None, epochs=1, **params):
        self.config = {**dict(optimizer=optimizer, noise=noise, batch_size=batch_size,
                              callbacks=callbacks, include_experiment_callbacks=include_experiment_callbacks,
                              steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, epochs=epochs),
                       **params}
        self.experiment = experiment
        self.optimizer = optimizer
        self._history = None
        self.noise = noise
        self.randomize = randomize

        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps

        self.epochs = epochs
        self.batch_size = batch_size
        if include_experiment_callbacks:
            self.callbacks = experiment.get_exp_callbacks() + (callbacks or [])
        else:
            self.callbacks = callbacks

        self.params = params or {}
        # todo: warn about extra params

    @staticmethod
    def ind_generator(num_samples, randomize=True):
        f = np.random.permutation if randomize else range
        while True:
            for i in f(num_samples):
                yield i

    def batch_generator(self, data=None, batch_size=2048, use_raw_data=True, mode='train', randomize=None):
        if randomize is None:
            randomize = self.randomize

        data = data or self.experiment.data

        inputs_outputs = data.raw_dataset if use_raw_data else ((data.train_x, data.train_y), (data.val_x, data.val_y))
        inputs_outputs = inputs_outputs[{'train': 0, 'val': 1}[mode]]

        num_samples = data.train_size if mode == 'train' else data.val_size

        ind_gen = self.ind_generator(num_samples=num_samples, randomize=randomize)

        batch_x = np.ndarray((batch_size,) + inputs_outputs[0].shape[1:])
        batch_y = np.ndarray((batch_size,) + inputs_outputs[1].shape[1:])

        while True:
            for a in [batch_x, batch_y]:
                if a is None:
                    continue
                a[:] = 0

            for i, ind in enumerate(ind_gen):
                batch_x[i] = inputs_outputs[0][ind]
                # todo: add noise
                batch_y[i] = inputs_outputs[1][ind]

                if i == (batch_size - 1):
                    break

            yield batch_x, batch_y

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
        data = exp.data
        model = exp.model

        batch_size = self.batch_size

        model.compile(self.optimizer, loss.loss_function, metrics)

        train_gen = self.batch_generator(mode='train')
        val_gen = self.batch_generator(mode='val')

        steps_per_epoch = self.steps_per_epoch or (data.train_size // batch_size)
        validation_steps = self.validation_steps or (data.val_size // batch_size)

        self._history = model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch,
                                            callbacks=self.callbacks, epochs=self.epochs,
                                            validation_data=val_gen, validation_steps=validation_steps, **self.params)

