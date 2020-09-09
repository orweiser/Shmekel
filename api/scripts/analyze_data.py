import argparse
import os
import sys
import pandas
import numpy as np
sys.path.insert(0, os.path.join('..', '..', '..', __file__))
from api.core import get_exp_from_config, load_config

# params = dict(dataset='StocksDataset', time_sample_length=5,
#               feature_list=[["SMA", {"range": 10}], ["SMA", {"range": 25}], ["SMA", {"range": 50}],
#                             ["RSI", {"range": 14}], ["RSI", {"range": 21}]],
#               stock_name_list=['fb'], years=[2012, 2013, 2014], val_mode=False)
# train_dataset = StocksDataset(**params)
# print("train dataset dates:")
# for date in train_dataset:
#     print(date['DateTuple'])
# params = dict(dataset='StocksDataset', time_sample_length=5,
#               feature_list=[["SMA", {"range": 10}], ["SMA", {"range": 25}], ["SMA", {"range": 50}],
#                             ["RSI", {"range": 14}], ["RSI", {"range": 21}]],
#               stock_name_list=['fb'], years=[2015, 2016, 2017], val_mode=True)
# val_dataset = StocksDataset(**params)
# print("val dataset dates:")
# for date in val_dataset:
#     print(date['DateTuple'])

def main(path):
    config = load_config(path)
    exp = get_exp_from_config(config)

    df = pandas.DataFrame(list(exp.val_dataset))
    x = np.array(list(df.inputs))
    y = np.array(list(df.outputs))
    exp.trainer.compile()

    # TODO run evaluate for each stock separately and return list of stocks and evaluation for each
    evaluation = exp.model.evaluate(x, y)
    if True:
        x = 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('config_path')
    args = parser.parse_args()

    main(args.config_path)
