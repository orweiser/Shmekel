from .keras_loss import KerasLoss
from .classification_reinforce import ClassificationReinforce


def get(loss: str, **kwargs):
    if loss == 'ClassificationReinforce':
        return ClassificationReinforce(loss=loss, **kwargs)

    return KerasLoss(loss=loss, **kwargs)
