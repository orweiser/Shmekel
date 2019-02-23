from copy import deepcopy as copy


class Loss:
    def __init__(self, experiment=None, **params):
        self.experiment = experiment

        self._config = copy(self.get_default_config())
        self._config.update(dict(experiment=experiment, **copy(params)))

        self.init(**params)

    def init(self, *args, **kwargs):
        raise NotImplementedError()

    def get_default_config(self) -> dict:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()

    def loss(self, y_true, y_pred):
        raise NotImplementedError()

    def __call__(self, y_true, y_pred):
        self.y_true_tensor = y_true
        self.y_pred_tensor = y_pred
        self.loss_tensor = self.loss(y_true, y_pred)

        return self.loss_tensor

    @property
    def callbacks(self):
        # todo
        return []
