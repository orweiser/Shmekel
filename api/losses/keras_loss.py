from ..core.loss import Loss
from keras.losses import get


class KerasLoss(Loss):
    loss_identifier: str

    def init(self):
        loss = self.config['loss']
        self.loss_identifier = loss

    def get_default_config(self) -> dict:
        return dict()

    def __str__(self) -> str:
        if isinstance(self.loss_identifier, str):
            return self.loss_identifier
        return str(get(self.loss_identifier))

    def loss(self, y_true, y_pred):
        return get(self.loss_identifier)(y_true, y_pred)
