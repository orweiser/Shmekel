from .keras_loss import KerasLoss
from .classification_reinforce import ClassificationReinforce
from utils.logger import logger


@logger.info_dec
def get(loss: str, **kwargs):
    if loss == 'ClassificationReinforce':
        return ClassificationReinforce(loss=loss, **kwargs)

    return KerasLoss(loss=loss, **kwargs)
