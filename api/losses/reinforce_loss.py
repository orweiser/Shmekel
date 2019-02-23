from .loss import Loss


class ReinforceLoss(Loss):
    def __init__(self, **kwargs):
        super(ReinforceLoss, self).__init__(kwargs)

    def init(self, rewards=(-1, 0), log_scale=True, minus=True):
        self._rewards = rewards
        self.minus = minus
        self.log_scale = log_scale

    @property
    def rewards(self):
        # todo: implement standardizing
        return self._rewards

    def loss_function(self, y_true, y_pred):
        pass

    @property
    def hyper_parameters(self):
        return self.rewards


