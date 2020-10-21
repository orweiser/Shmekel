from api.models import get as get_model
import numpy as np
import Trader.utilities.sandbox as sb
import Trader.Core.Candle as C
import json
import random


def get_open(candle):
    return candle.open


def get_close(candle):
    return candle.close


def get_high(candle):
    return candle.high


def get_low(candle):
    return candle.low


def get_sma_10(candle):
    return candle.sma_10


def get_sma_25(candle):
    return candle.sma_25


def get_sma_50(candle):
    return candle.sma_50


def get_rsi_14(candle):
    return candle.rsi_14


def get_rsi_21(candle):
    return candle.rsi_21


INPUTS = {
    "open": get_open,
    "close": get_close,
    "high": get_high,
    "low": get_low,
    "SMA_10": get_sma_10,
    "SMA_25": get_sma_25,
    "SMA_50": get_sma_50,
    "RSI_14": get_rsi_14,
    "RSI_21": get_rsi_21
}
OUTPUTS = ["rise"]


def read_json_path(path):
    with open(path) as f:
        return json.load(f)


def get_model_from_config(config):
    model = get_model(**config['model'])
    model.load_weights(config['weights_path'])
    return model


class Predictor:
    def __init__(self, cfg_path):
        self.cfg = read_json_path(cfg_path)
        self.model = get_model_from_config(self.cfg)
        self.candles = {}

    def get_time_sample_length(self):
        return self.cfg["dataset"]["time_sample_length"]

    def get_model(self):
        return self.model

    def set_candles(self, candles):
        for i in range(self.cfg["dataset"]["time_sample_length"]):
            for input_feature in self.cfg["dataset"]["input_features"]:
                if "range" in input_feature:
                    input_feature = "_".join([input_feature["type"], str(input_feature["range"])])
                else:
                    input_feature = input_feature["type"]
                if input_feature not in self.candles:
                    self.candles[input_feature] = []
                self.candles[input_feature].append(INPUTS[input_feature](candles[-(i + 1)]))
        return self.get_prediction()

    def new_candle(self, candle):
        for input_feature in self.cfg["dataset"]["input_features"]:
            if "range" in input_feature:
                input_feature = "_".join([input_feature["type"], str(input_feature["range"])])
            else:
                input_feature = input_feature["type"]
            self.candles[input_feature] = [INPUTS[input_feature](candle)] + self.candles[input_feature][0:-1]
        return self.get_prediction()

    def get_prediction(self):
        for key in self.candles.keys():
            data = np.array([[v for v in self.candles[key]] for key in self.candles.keys()]).T
        predicts = self.model.predict_on_batch(np.reshape(data, (1,) + data.shape))
        # out_predicts = {out: [] for out in OUTPUTS}
        out_predicts = {}
        for i, out in enumerate(self.cfg["dataset"]["output_features"]):
            if out["type"] not in out_predicts:
                out_predicts[out["type"]] = []
            if ["output_type"] == "categorical":
                out_predicts[out["type"]].append((out["k_next_candle"], predicts[i][1]))
            else:
                out_predicts[out["type"]].append((out["k_next_candle"], predicts[i]))

        return out_predicts  # sb.((out, out_predicts[out]) for out in OUTPUTS)


def random_candle():
    return C.Candle(random.random(), random.random(), random.random(), random.random(), random.random(),
                    random.random(), random.random(), random.random(), random.random(), random.random(),
                    random.random(), random.random())


# testing code above
my_predicctor = Predictor("F:\\Users\\Ron\\Shmekels\\regression_for_maor\\exported_configs\\exported_500.json")
candels = []
for i in range(my_predicctor.get_time_sample_length()):
    candels.append(random_candle())
print(my_predicctor.set_candles(candels))
print(my_predicctor.new_candle(random_candle()))
