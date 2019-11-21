from api.datasets.stock_dataset import StocksDataset
from feature_space import mapping
from api.core.benchmarker import Benchmarker


iter_stock = Benchmarker.iter_stock


params = dict(normalization_type=None)
dataset = StocksDataset('val', val_mode=True, feature_list=[('Candle', params)],
                        output_feature_list=[(key, params) for key in mapping])

for stock in dataset.stocks_list:

    stock_name = stock.stock_tckt

    for sample in iter_stock(dataset, stock):
        date = sample['DateTuple']
        raw_candle = sample['RawCandle']

        outputs = None





