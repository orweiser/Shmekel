from api.core.trader import Trader
from api.datasets import StocksDataset


class Benchmarker:
    def __init__(self, experiment, traders=None):
        self.experiment = experiment
        self.traders = [self._parse_trader(t) for t in traders]

    @staticmethod
    def _parse_trader(trader):
        pass

    def train_trader(self, trader):
        pass

    def to_csvs(self, mode='val'):
        pass






