from ..core import Experiment
import numpy as np
from ..datasets import StocksDataset
from itertools import product
from time import sleep
from copy import deepcopy
from api.tests.dataset_tests import _get_dataset_params, ORIGINAL_TEST_CASES


class TestStockDataset:

    def __init__(self):
        pass

    def get_experiment(self, with_backup=False, **kwargs) -> Experiment:
        params=dict(
            name='test_experiment',
            model_config=None, loss_config=None,
            train_dataset_config=None, val_dataset_config=None,
            train_config = None, backup_config = None, metrics_list = None
        )
        exp = Experiment(**kwargs)
        raise NotImplementedError()

    def test_augmentations(self):
        exp = self.get_experiment()
        for gen in [exp.trainer.get_batch_generator(mode) for mode in ['train', 'val']]:
            pass

    def run_all(self):
        logs = []
        for k in dir(self):
            if k.startswith('test'):
                try:
                    getattr(self, k)()
                except Exception as e:
                    logs.append('FAILED: {0} \t\t {1}'.format(k, getattr(e, 'message', '')))
                else:
                    logs.append('PASSED: %s' % k)
        sleep(0.001)
        print(*logs, sep='\n')





