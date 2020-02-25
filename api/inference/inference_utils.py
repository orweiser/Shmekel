import sys
import os
sys.path.append(os.getcwd())
from api.datasets.stock_dataset import InferenceStocksDataset
from api.models import get as get_model
import json
from api.core.trader import Trader
from api.datasets import StocksDataset
from shmekel_core import Stock
from api.utils.data_utils import batch_generator
import os
from tqdm import tqdm
import numpy as np


def read_json_path(path):
    with open(path) as f:
        return json.load(f)


def get_model_from_config(config):
    model = get_model(**config['model'])
    model.load_weights(config['weights_path'])
    return model


def get_dataset(config, folder):
    return InferenceStocksDataset(basedir=folder, **config['dataset'])


def main_(config, folder, out_folder):
    model = get_model_from_config(config)
    dataset = get_dataset(config, folder)
    predict_on_dataset(dataset, model, out_folder)


def main(exported_config_path, folder, out_folder=None):
    config = read_json_path(exported_config_path)
    main_(config, folder, out_folder)


def flatten_dictionaries(dict_list):
    out = {key: [] for key in dict_list[0]}
    for d in dict_list:
        for key in d:
            out[key].append(d[key])
    return out


title = ('Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'prediction')


def dump_to_csv(path, dict):
    raw_candle = np.array(dict['RawCandle'])
    new_dict = {
        'Open': raw_candle[:, 0],
        'High': raw_candle[:, 1],
        'Low': raw_candle[:, 2],
        'Close': raw_candle[:, 3],
        'Volume': raw_candle[:, 4],
        'Date': ['/'.join(map(str, date)) for date in dict['DateTuple']],
        'prediction': np.array(dict['prediction'])[:, 1]}
    data = np.array([[v for v in new_dict[key]] for key in title]).T
    data = [','.join([str(x) for x in line]) for line in [title] + list(data)]
    with open(path, 'w') as f:
        f.write('\n'.join(data))


def predict_on_dataset(dataset, model, out_folder, batch_size=512):
    generator = batch_generator(dataset, batch_size=batch_size, randomize=False,
                                ind_gen=(i for i in range(len(dataset))))

    predictions = []
    for batch_in, batch_out in tqdm(generator):
        predictions.extend(list(model.predict_on_batch(batch_in)))

    assert len(predictions) == len(dataset), '{}  {}'.format(len(predictions), len(dataset))
    csv_data = {}
    for pred, sample in zip(predictions, dataset):
        sample['prediction'] = pred
        csv_data.setdefault(sample['stock'], []).append(sample)
    csv_data = {key: flatten_dictionaries(val) for key, val in csv_data.items()}

    for stock, dict in csv_data.items():
        dump_to_csv(os.path.join(out_folder, stock.stock_tckt + '.csv'), dict)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(usage="This module performs inference on stock data\n"
                                           "inference_utils.py -c <config_path> -i <input_dir> -o <output_dir>",
                                     prog="inference_utils.py",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config", dest="config_path", help="Path to a Shmekel config file", required=True)
    parser.add_argument("-i", "--input", dest="input_dir", help="Path to input folder", required=True)
    parser.add_argument("-o", "--output", dest="output_dir", help="Path to output folder", required=True)

    args = parser.parse_args()

    main(args.config_path, args.input_dir, args.output_dir)
