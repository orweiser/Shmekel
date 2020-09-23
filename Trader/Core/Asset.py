import os
from random import uniform

from Trader.Core.enums import *
from feature_space_2020.SMA import *
from feature_space_2020.RSI import *
from Trader.Core.Candle import *

def initialize_predictors():
    print("initialize predictors")


class Asset:
    def __init__(self, name, asset_type, sector, candles=[]):
        self.name = name
        self.type = asset_type
        self.sector = sector
        self.candles = candles
        self.predictors = []

    def calculate_features(self):
        rsi_ranges = [14, 21]
        sma_ranges = [10, 25, 50]
        # open = [o.open for o in self.candles]
        # high = [o.high for o in self.candles]
        # low = [o.low for o in self.candles]
        close = [o.close for o in self.candles]

        for rsi_range in rsi_ranges:
            rsi = RSI()
            rsi.range = rsi_range
            rsi.auto_fill = True
            rsi_out = rsi.process(close)

            for i, candle in enumerate(self.candles):
                candle.features["RSI_" + str(rsi_range)] = rsi_out[i]

        for sma_range in sma_ranges:
            sma = SMA()
            sma.range = sma_range
            sma.auto_fill = True
            sma_out = sma.process(close)

            for i, candle in enumerate(self.candles):
                candle.features["SMA_" + str(sma_range)] = sma_out[i]

        print(self.candles[0].features)

    def load_predictors(self, runmode):
        if runmode == RunMode.simulate:
            self.candles = fake_predict_candles(self.candles)

    def simulate(self,trade_config):
        print("simulation start for Asset: "+self.name)
        for i, candle in enumerate(self.candles):
            avg_rise = get_avg_prediction_by_type(candle.predictions)
            print(candle.predictions)


def get_avg_prediction_by_type(predictors = {}):
    for key, value in predictors.items():
        print(key, '->', value)

def fake_predict_candles(candles):
    fake_ranges = [1, 2, 3]
    fake_predict_types = [PredictionType.rise,
                          PredictionType.close_to_open_ratio,
                          PredictionType.high_to_open_ratio,
                          PredictionType.low_to_open_ratio]

    def get_fake_value_by_type(fake_type):
        if fake_type == PredictionType.rise:
            return uniform(0, 1)
        if fake_type == PredictionType.high_to_open_ratio:
            return uniform(0, 0.1)
        if fake_type == PredictionType.close_to_open_ratio:
            return uniform(-0.1, 0.1)
        if fake_type == PredictionType.low_to_open_ratio:
            return uniform(-0.1, 0)

    for candle in candles:
        for range in fake_ranges:
            for type in fake_predict_types:
                fake_prediction = Fake_Prediction(type, range)
                fake_prediction.value = get_fake_value_by_type(type)
                fake_id = str(type)+"_"+str(range)
                candle.predictions[fake_id] = fake_prediction
                # print("fake value: "+str(fake_prediction.value))
    return candles


def load_by_exported_config():
    dir_name = Location.models_dir
    files = get_list_of_files(dir_name)
    for file in files:
        print(file)


def get_list_of_files(dir_name):
    file_list = os.listdir(dir_name)
    all_files = list()
    for entry in file_list:
        full_path = os.path.join(dir_name, entry)
        if os.path.isdir(full_path):
            all_files = all_files + get_list_of_files(full_path)
        else:
            all_files.append(full_path)

    return all_files
