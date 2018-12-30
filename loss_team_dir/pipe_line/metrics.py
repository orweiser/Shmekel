import keras.backend as K
from tensorflow import where
from keras.metrics import categorical_accuracy


def get_metrics(loss_name='categorical_crossentropy'):
    metrics = ['acc', total_sharpness]
    if loss_name != 'categorical_crossentropy':
        metrics += [
            uncertain_fraction,
            certain_predictions_acc,
            certainty_sharpness,
            uncertainty_sharpness
        ]
    return metrics


def _certain_ind(y_pred, uncertain=False):
    if uncertain:
        return where(K.equal(
            K.max(y_pred, -1),
            y_pred[:, -1])
        )
    else:
        return where(K.not_equal(
            K.max(y_pred, -1),
            y_pred[:, -1])
        )


def uncertain_fraction(y_true, y_pred):
    return K.cast(
        K.equal(K.max(y_pred, -1), y_pred[:, -1])
        , 'float32')


def certain_predictions_acc(y_true, y_pred):
    c_pred_ind = _certain_ind(y_pred, uncertain=False)

    return categorical_accuracy(
        K.gather(y_true, c_pred_ind),
        K.gather(y_pred, c_pred_ind)
    )


def certainty_sharpness(y_true, y_pred):
    c_pred_ind = _certain_ind(y_pred, uncertain=False)

    return K.max(K.gather(y_pred, c_pred_ind), axis=-1)


def uncertainty_sharpness(y_true, y_pred):
    uc_pred_ind = _certain_ind(y_pred, uncertain=True)

    return K.max(K.gather(y_pred, uc_pred_ind), axis=-1)


def total_sharpness(y_true, y_pred):
    return K.max(y_pred, axis=-1)
