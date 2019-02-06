

class Loss:
    def __init__(self, experiment=None, **params):
        self.config = {**dict(), **params}

        self.experiment = experiment
        self._tensor_loss = None

    def loss_function(self, y_true, y_pred):
        raise NotImplementedError

    def __call__(self, y_true, y_pred):
        self.tensor_loss = self.loss_function(y_true, y_pred)
        return self.tensor_loss

    @property
    def hyper_parameters(self):
        raise NotImplementedError

    @property
    def tensor_hyper_parameters(self):
        raise NotImplementedError

    @property
    def tensor_loss(self):
        return self._tensor_loss

    @tensor_loss.setter
    def tensor_loss(self, value):
        if self._tensor_loss is not None:
            raise RuntimeError("setting tensor_loss twice is not allowed")
        self._tensor_loss = value

    @property
    def callbacks(self):
        # todo
        return []
