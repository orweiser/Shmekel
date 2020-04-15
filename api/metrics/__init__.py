from Utils.logger import logger
import keras.backend as K
from tensorflow import where
from keras.metrics import categorical_accuracy


def uncertainty_sharpness(y_true, y_pred):
    """
    sharpness of prediction of uncertain predictions
    """
    uc_pred_ind = _certain_ind(y_pred, uncertain=True)

    return K.max(K.gather(y_pred, uc_pred_ind), axis=-1)


def uncertain_fraction(y_true, y_pred):
    """
    fraction of predictions that are uncertain
    """
    return K.cast(
        K.equal(K.max(y_pred, -1), y_pred[:, -1])
        , 'float32')


def certain_predictions_acc(y_true, y_pred):
    """
    accuracy of predictions that are certain
    """
    c_pred_ind = _certain_ind(y_pred, uncertain=False)

    return categorical_accuracy(
        K.gather(y_true, c_pred_ind),
        K.gather(y_pred, c_pred_ind)
    )


def certainty_sharpness(y_true, y_pred):
    """
    sharpness of prediction of certain predictions
    """
    c_pred_ind = _certain_ind(y_pred, uncertain=False)

    return K.max(K.gather(y_pred, c_pred_ind), axis=-1)


def total_sharpness(y_true, y_pred):
    """sharpness of prediction"""
    return K.max(y_pred, axis=-1)


metrics_mapping = {
    'UncertaintySharpness': uncertainty_sharpness,
    'UncertainFraction': uncertain_fraction,
    'CertainPredictionAcc': certain_predictions_acc,
    'CertaintySharpness': certainty_sharpness,
    'TotalSharpness': total_sharpness,
}


@logger.info_dec
def get(requested_metrics, **kwargs):
    metrics = ['acc']
    for metric in requested_metrics or []:
        if metric in metrics:
            continue
        if metric in metrics_mapping:
            metrics.append(metrics_mapping[metric])
        else:
            metrics.append(metric)
    return metrics


def _certain_ind(y_pred, uncertain=False):
    """
    returns a tensor of indices where predictions are certain\certain, according to :param y_pred
    """
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
