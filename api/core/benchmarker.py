from api.core.trader import Trader
from api.datasets import StocksDataset
from shmekel_core import Stock
from api.utils.data_utils import batch_generator


class Benchmarker:
    def __init__(self, experiment):
        self.experiment = experiment

    def get_dataset(self, mode='val'):
        return getattr(self.experiment, '%s_dataset' % mode)

    def find_indices(self, stock: Stock, mode='val'):
        dataset = self.get_dataset(mode)
        for i in range(len(dataset)):
            if dataset.stock_and_local_index_from_global_index(index=i)[1] is stock:
                yield i

    def iter_stock(self, stock: Stock, mode='val'):
        dataset = self.get_dataset(mode)
        for i in self.find_indices(stock, mode):
            yield dataset[i]

    def predict_on_stock(self, stock: Stock, mode='val') -> dict:
        dataset = self.get_dataset(mode)
        generator = batch_generator(dataset, batch_size=self.experiment.trainer.batch_size, randomize=False,
                                    augmentations=self.experiment.trainer.augmentations['val'],
                                    ind_gen=self.iter_stock(stock, mode))

        predictions = []
        for batch_in, batch_out in generator:
            # todo: predict all batches
            raise NotImplementedError()
        # concat predictions

        data = {}
        generator = batch_generator(dataset, batch_size=self.experiment.trainer.batch_size, randomize=False,
                                    augmentations=self.experiment.trainer.augmentations['val'],
                                    ind_gen=self.iter_stock(stock, mode))
        for i, p in zip(generator, predictions):
            raw_data = self.get_raw_data(i, dataset)  # dict with open, high low close, volume, date

            for key, val in raw_data.items():
                data.setdefault(key, []).append(val)

            data.setdefault('prediction', p)
        return data

        """ 
        returns dict = {
            Open: [...]
            High: [...]
            low: [...]
            close: [...]
            volume: [...]
            prediction: [...]
            }
        """
        raise NotImplementedError()

    def dump_prediction_to_csv(self, prediction: dict, path):
        raise NotImplementedError()

    def call_trader_on_stock(self, stock, mode)->dict:
        """
        todo:
         prediction = self.predict on stock
         dump_prediction(csv_path)
         run exe
         performances = load_outputs (summary)
         return performances
         """
        raise NotImplementedError

    @staticmethod
    def get_raw_data(index, dataset):
        item = dataset[index]
        date, candle = [item[key] for key in ['DateTuple', 'RawCandle']]

        raw_data = {key: val for key, val in zip(['open', 'low', 'high', 'close', 'volume'], candle)}
        raw_data['date'] = date

        return raw_data

    def run_on_dataset(self, dataset):
        outputs = {}
        for stock in dataset.stocks_list:
            # todo: run on stock, load summary, add to outputs
            pass
        return outputs

    def evaluate(self, mode='val'):
        # todo: run on dataset, get all outputs, do evaluation ?? and save
        raise NotImplementedError

    # todo: train traders
