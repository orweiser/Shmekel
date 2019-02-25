

class Loss:
    def __init__(self, experiment=None, with_uncertainty=False, **params):
        self.config = {'experiment': experiment, 'with_uncertainty': with_uncertainty, **params}

        self.with_uncertainty = with_uncertainty
        self.experiment = experiment
        self._tensor_loss = None

        if hasattr(self, 'init'):
            self.init(**params)

    def loss_function(self, y_true, y_pred):
        raise NotImplementedError

    def __call__(self, y_true, y_pred):
        if self.with_uncertainty:
            if y_true.output:
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
