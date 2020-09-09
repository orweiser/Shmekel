from api.models import get as get_model
import numpy as np
import Trader.utilities.sandbox as sb
import json

INPUTS = {
    "open":  open,
    "close": close,
    "high":  high,
    "low":   low
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
        return self.cfg["time sample length"]

    def get_model(self):
        return self.model

    def set_candles(self, candles):
        for i in range(self.cfg["time sample length"]):
            for input_feature in self.cfg["input features"]:
                self.candles[input_feature].append(candles[-(i+1)].INPUTS[input_feature])
        return self.get_prediction()

    def new_candle(self, candle):
        for input_feature in self.cfg["input features"]:
            self.candles[input_feature] = [candle.INPUTS[input_feature]] + self.candles[input][0:-1]
        return self.get_prediction()

    def get_prediction(self):
        data = np.array([[v for v in self.candles[key]] for key in self.cfg["input features"]]).T
        predicts = self.model.predict_on_batch(data)
        out_predicts = {out: [] for out in OUTPUTS}
        for i, out in enumerate(self.cfg["dataset"]["output_features"]):
            if out[1]["output_type"] == "categorical":
                out_predicts[out[0]].append((out[1]["k_next_candle"], predicts[i][1]))
            else:
                out_predicts[out[0]].append((out[1]["k_next_candle"], predicts[i]))
        return [Prediction(out, out_predicts[out] for out in OUTPUTS]