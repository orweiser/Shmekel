from api.datasets.stock_dataset import InferenceStocksDataset
from api.models import get as get_model
import json


def read_json_path(path):
    with open(path) as f:
        return json.load(f)


def get_model_from_config(config):
    model = get_model(**config['model'])
    model.load_weights(config['weights_path'])
    return model


def get_dataset(config, folder):
    return InferenceStocksDataset(basedir=folder, **config['dataset'])


def get_model_and_dataset(exported_config_path, folder):
    config = read_json_path(exported_config_path)

    model = get_model_from_config(config)
    dataset = get_dataset(config, folder)

    return model, dataset


def predict_on_item(model, dataset, index):
    sample = dataset[index]
    x = sample['inputs'][None]
    sample['prediction'] = model.predict(x)
    return sample

