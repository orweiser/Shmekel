from .loss import Loss


class KerasLoss(Loss):
    def __init__(self, loss, **params):
        super(KerasLoss, self).__init__(**params)

        from keras.losses import get
        self._loss_func = get(loss)

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


