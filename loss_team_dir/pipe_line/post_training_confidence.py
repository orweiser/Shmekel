from keras import Model
import numpy as np


def get_intervals_accuracy(model, train_x, train_y, num_intervals=100):
    """
    :type model: Model
    :param model:
    :param x:
    :param y:
    :param interval:
    :return:
    """
    pred = model.predict(train_x)
    max_p = pred.max(axis=-1)
    pred = pred.argmax(axis=-1)

    cnt = np.zeros((num_intervals,))
    probs = np.zeros((num_intervals,))

    for i, (m, p, y) in enumerate(zip(max_p, pred, train_y.argmax(-1))):
        ind = (m * num_intervals) // 1
        ind = ind if ind < num_intervals else (ind - 1)
        ind = int(ind)

        cnt[ind] += 1
        probs[ind] += 1 * (p == y)

    probs[cnt > 0] /= cnt[cnt > 0]

    return probs, cnt
