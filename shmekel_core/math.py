"""
TODO: Add doc string for this file
"""
import numpy as np

EPS_DOUBLE = np.finfo(float).eps
EPS_FLOAT = np.finfo(np.float32).eps


def smooth_moving_avg(data_seq, period):
    data_seq = np.flip(data_seq)
    smma = np.ndarray(np.size(data_seq) - period + 1)
    smma[0] = np.mean(data_seq[:period])

    for idx in range(period, np.size(data_seq)):
        smma[idx - period + 1] = (1 - 1/period)*smma[idx - period] + (1/period)*data_seq[idx]

    smma = np.flip(smma)
    return smma


def exponential_moving_average(data_seq, period):
    ma = smooth_moving_avg(data_seq, period)
    mult = 2. / (period + 1)
    ema = np.zeros(ma.shape)
    ema[-1] = data_seq[len(ma) - 1] * mult
    for i in range(2, len(ma)):
        ema[-i] = data_seq[len(ma) - i] * ema[-(i + 1)] * (1 - mult)

    return ema
