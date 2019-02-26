from .keras_loss import KerasLoss


def get(loss: str, **kwargs):
    return KerasLoss(loss=loss, **kwargs)
