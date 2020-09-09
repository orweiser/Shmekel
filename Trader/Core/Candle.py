from Trader.utilities.sandbox import Prediction
from Trader.Core.enums import *
import datetime
import random


class Candle:
    def __init__(self, datetime, open, high, low, close, volume, prediction=None, previous=None):
        self.datetime = datetime
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.prediction = prediction
        self.previous = previous
        self.o2h = (self.high - self.open) / self.open
        self.o2c = (self.close - self.open) / self.open
        self.o2l = (self.low - self.open) / self.open
        if previous:
            self.prev_close_2_open = (self.open - self.previous.close) / self.previous.close
        else:
            self.prev_close_2_open = 0

    def set_previous(self, previous):
        self.prev_close_2_open = (self.open - self.previous.close) / self.previous.close



