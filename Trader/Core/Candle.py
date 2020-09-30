from Trader.utilities.sandbox import Prediction
from Trader.Core.enums import *
import datetime
import random


class Candle:
    def __init__(self, datetime, open, high, low, close, volume, sma_10, sma_25, sma_50, rsi_14, rsi_21, prediction=None, previous=None):
        self.datetime = datetime
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.prediction = prediction
        self.previous = previous
        self.sma_10 = sma_10
        self.sma_25 = sma_25
        self.sma_50 = sma_50
        self.rsi_14 = rsi_14
        self.rsi_21 = rsi_21
        self.o2h = (self.high - self.open) / self.open
        self.o2c = (self.close - self.open) / self.open
        self.o2l = (self.low - self.open) / self.open
        if previous:
            self.prev_close_2_open = (self.open - self.previous.close) / self.previous.close
        else:
            self.prev_close_2_open = 0

    def set_previous(self, previous):
        self.prev_close_2_open = (self.open - self.previous.close) / self.previous.close

    def fake_predict(self):
        fake_input = []
        k_array = [1, 2, 3, 4, 5, 10]
        for k in k_array:
            val = 1 / random.randrange(1, 10)
            fake_input.append((k, val))
        prediction_type = PredictionType.rise
        self.prediction = Prediction(fake_input, prediction_type)


# time = datetime.datetime.now()
#
# myCandle = Candle(time, 95.1, 98.7, 94.6, 97.2, 1000)
# myCandle.fake_predict()

