from .loss import Loss
from .reinforce_loss import ReinforceLoss


class KerasLoss(Loss):
    def __init__(self, loss, **params):
        super(KerasLoss, self).__init__(**params)
        self.config.update(dict(loss=loss))

        from keras.losses import get as get_keras_loss
        self._loss_func = get_keras_loss(loss)

    def loss_function(self, y_true, y_pred):
        return self._loss_func(y_true, y_pred)


def get(loss, **params):
    # if issubclass(Loss, loss):
    #     return loss(**params)
    #
    # if isinstance(loss, Loss):
    #     if params:
    #         print('Warning: unexpected params when loss is an instance of Loss')
    #     return loss

    if isinstance(loss, str):
        pass

    return KerasLoss(loss, **params)


