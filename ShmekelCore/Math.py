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