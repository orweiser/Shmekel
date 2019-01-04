import keras.backend as K
from tensorflow import where
from keras.metrics import categorical_accuracy


def get_metrics(without_uncertainty=False):
    """
    get a list of metrics for experiment
    :param without_uncertainty: boolean, if True, there is no uncertainty class in the classification task
    :return: a list of metrics
        if without_uncertainty:
            metrics are: accuracy, total_sharpness
        else:
            metrics are: accuracy, total_sharpness, uncertain_fraction, certain_predictions_acc,
                        certainty_sharpness, uncertainty_sharpness
    """
    metrics = ['acc', total_sharpness]
    if not without_uncertainty:
        metrics += [
            uncertain_fraction,
            certain_predictions_acc,
            certainty_sharpness,
            uncertainty_sharpness
        ]
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


def uncertainty_sharpness(y_true, y_pred):
    """
    sharpness of prediction of uncertain predictions
    """
    uc_pred_ind = _certain_ind(y_pred, uncertain=True)

    return K.max(K.gather(y_pred, uc_pred_ind), axis=-1)


def total_sharpness(y_true, y_pred):
    """sharpness of prediction"""
    return K.max(y_pred, axis=-1)
