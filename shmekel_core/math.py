"""
TODO: Add doc string for this file
"""
import numpy as np

EPS_DOUBLE = np.finfo(float).eps
EPS_FLOAT = np.finfo(np.float32).eps


def smooth_moving_avg(data_seq, period):
    data_seq = np.flip(data_seq, axis=0)
    smma = np.ndarray(np.size(data_seq) - period + 1)
    smma[0] = np.mean(data_seq[:period])

    for idx in range(period, np.size(data_seq)):
        smma[idx - period + 1] = (1 - 1/period)*smma[idx - period] + (1/period)*data_seq[idx]

    smma = np.flip(smma, axis=0)
    return smma


def exponential_moving_average(data_seq, period):
    ma = smooth_moving_avg(data_seq, period)
    mult = 2. / (period + 1)
    ema = np.zeros(ma.shape)
    ema[-1] = data_seq[len(ma) - 1] * mult
    for i in range(2, len(ma)):
        ema[-i] = data_seq[len(ma) - i] * ema[-(i + 1)] * (1 - mult)

    return ema


def smooth_moving_avg_investopedia(data_seq, period):
    sma = np.zeros(data_seq.shape[0] - period + 1)
    sma_size = np.size(sma)
    for i in range(period):
        sma = sma + data_seq[i: i + sma_size]

    sma = sma / period
    return sma


def exponential_moving_average_investopedia(data_seq, period):
    Smoothing = 2
    ema_factor = Smoothing / (1 + period)
    ema = np.zeros(data_seq.shape[0] - period + 1)
    ema[0] = np.sum(data_seq[:period]) / period  # initialize first e
    for i in range(1, len(ema)):
        ema[i] = data_seq[i+period-1]*ema_factor + ema[i-1]*(1 - ema_factor)

    return ema