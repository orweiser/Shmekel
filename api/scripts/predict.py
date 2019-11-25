import os
import sys
import numpy as np
from glob import glob
from keras import Model
from keras.models import model_from_json

from shmekel_core import Stock


def parse_argv(argv):
    weights_path, arch_path, input_dir, output_dir, feature_list = argv
    assert os.path.exists(weights_path), "Weights path does not exist!"
    assert os.path.exists(arch_path), "Arch path does not exist!"
    assert os.path.isdir(input_dir), "Input dir does not exist!"
    assert os.path.exists(output_dir), "Output sir does not exist!"
    feature_list = feature_list if feature_list else ('High', 'Open', 'Low', 'Close', 'Volume')
    return weights_path, arch_path, input_dir, output_dir, feature_list


def read_a_file(file_path):
    with open(file_path) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    return [x.strip() for x in content]


def file_to_dict(file_path):
    content = read_a_file(file_path)
    if not content:
        return False

    keys = content[0].split(',')
    log = {key: [] for key in keys}
    for line in content[1:]:
        for key, val in zip(keys, line.split(',')):
            if key == 'Date':
                log[key].append(tuple([int(v) for v in val.split('-')]))
            else:
                log[key].append(float(val))
    return log


def load_stock(file_path, pattern=('Open', 'High', 'Low', 'Close', 'Volume')):
    log = file_to_dict(file_path)

    if not log:
        return log

    a = np.stack([log[key] for key in pattern])

    return a, log['Date']


def get_stock_paths(input_dir):
    return glob(os.path.join(input_dir, "*"))


def get_stock_name(stock_path):
    base_name = os.path.basename(stock_path)
    return os.path.splitext(base_name)[0]


def create_stock_list(stock_paths, feature_list):
    stocks = []
    for stock_path in stock_paths:
        stocks.append(Stock(load_stock(stock_path), stock_tckt=get_stock_name(stock_path),
                            feature_list=feature_list))
    return stocks


def predict_all_stocks(model, stocks):
    all_results = []
    # todo: do we need to crop temporal slices for FF or just pass one day sample?
    for stock in stocks:
        results = []
        raw_data, date = stock.data
        feature_matrix = stock.feature_matrix
        for i in feature_matrix.shape[0]:
            results.append((date[i], raw_data[i], model.predict(feature_matrix[i])))
        all_results.append((stock.stock_tckt, results))
    return all_results


def dump_results(output_dir, all_results):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for stock_name, results in all_results:
        file_name = os.path.join(output_dir, stock_name, ".csv")
        with open(file_name) as f:
            for date, raw_data, prediction in results:
                f.write(",".join(str(s) for s in [date, ",".join(str(s) for s in raw_data),
                                                  prediction]))


def run(argv):
    assert len(argv) >= 4
    print("Parsing arguments given...")
    weights_path, arch_path, input_dir, output_dir, feature_list = parse_argv(argv)
    print("Creating Stock objects from ascii files...")
    stocks = create_stock_list(get_stock_paths(input_dir), feature_list)

    print("Loading model...")
    model = model_from_json(arch_path)
    model.load_weights(weights_path)

    print("Predicting...")
    all_results = predict_all_stocks(model, stocks)

    print("Dumping results...")
    dump_results(output_dir, all_results)


if __name__ == "__main__":
    run(sys.argv)

