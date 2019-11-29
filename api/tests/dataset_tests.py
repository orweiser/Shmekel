from api.utils.data_utils import batch_generator
import numpy as np
from ..datasets import StocksDataset
from itertools import product
from time import sleep
from copy import deepcopy


ORIGINAL_TEST_CASES = {
        0: {
            'inputs': np.array([[42.05,  45,     38,     38.23, 580438450],
                                [36.53,  36.66,  33,     34.03,  169418988]]),
            'output': np.array([1, 0]),
            'date': (2012, 5, 21),
            'stock': 'fb'
        },
        1: {
            'inputs': np.array([[36.53,  36.66,  33,     34.03,  169418988],
                                [32.61,  33.59,  30.94,  31,     101876406]]),
            'output': np.array([0, 1]),
            'date': (2012, 5, 22),
            'stock': 'fb'
        },
        1378: {
            'inputs': np.array([[179.79, 180.35, 179.11, 179.56, 10467606],
                                [178.31, 179.4, 177.09, 179.3, 12602188]]),
            'output': np.array([0, 1]),
            'date': (2017, 11, 9),
            'stock': 'fb'
        },
        1379: {
            'inputs': np.array([[28.247, 28.528, 27.978, 28.528, 490859],
                                [28.379, 28.666, 28.186, 28.264, 332954]]),
            'output': np.array([1, 0]),
            'date': (2005, 2, 28),
            'stock': 'ab'
        },
        1380: {
            'inputs': np.array([[28.379, 28.666, 28.186, 28.264, 332954],
                                [28.414, 28.637, 28.192, 28.254, 403173]]),
            'output': np.array([1, 0]),
            'date': (2005, 3, 1),
            'stock': 'ab'
        }
    }


def _get_dataset_params(candle_params=None, **kwargs):
    params = dict(time_sample_length=2,
                  feature_list=[('Candle', candle_params or {})],
                  stock_name_list=['fb', 'ab'], output_feature_list=[('Rise', {})])
    params.update(kwargs)
    return params


def get_test_dataset(candle_params=None, **kwargs):
    """

    config_path=None, time_sample_length=5,
         stock_name_list=None, feature_list=None, val_mode=False, output_feature_list=None,
         split_by_year=False
    :return:
    """
    params = _get_dataset_params(candle_params=candle_params, **kwargs)
    return StocksDataset('dataset', **params)


class TestStockDataset:

    def __init__(self):
        pass

    TEST_CASES = deepcopy(ORIGINAL_TEST_CASES)

    def test_samples(self):
        return self._test_samples(self.TEST_CASES, candle_params=0)

    def test_samples_pos_delay(self):
        test_cases = {
            0: self.TEST_CASES[1],
            1377: self.TEST_CASES[1378],
            1378: self.TEST_CASES[1380]
        }

        return self._test_samples(test_cases, candle_params={'time_delay': 1})

    def test_years(self):
        years_gen = (
            (2015, 2010, 2200, 1900, 2005),
            (),
            (1000,),
            [i for i in range(1980, 2020)]
        )

        had_samples = False
        for years in years_gen:
            dataset = get_test_dataset(years=years)
            if len(dataset):
                had_samples = True
        assert had_samples

        for years in years_gen:
            dataset = get_test_dataset(years=years)
            for i in range(len(dataset)):
                sample = dataset[i]
                assert sample['DateTuple'][0] in years

            try:
                _ = dataset[len(dataset)]
            except IndexError:
                pass
            else:
                raise AssertionError('Dataset has more samples than it reports')

    def _test_samples(self, test_cases, candle_params):
        dataset = get_test_dataset(candle_params)

        """
            Date,       Open,   High,   Low,    Close,  Volume, OpenInt
            2012-05-18, 42.05,  45,     38,     38.23,  580438450,  0
            2012-05-21, 36.53,  36.66,  33,     34.03,  169418988,  0
            2012-05-22, 32.61,  33.59,  30.94,  31,     101876406,  0
            2012-05-23, 31.37,  32.5,   31.36,  32,     73678512,   0
            
            2017-11-06,178.56,180.45,178.31,180.17,13275578,0
            2017-11-07,180.5,180.748,179.403,180.25,12903836,0
            2017-11-08,179.79,180.35,179.11,179.56,10467606,0
            2017-11-09,178.31,179.4,177.09,179.3,12602188,0
            2017-11-10,178.35,179.1,177.96,178.46,11060355,0

            Date       Open       High       Low       Close       Volume       OpenInt
            2005-02-25       28.247       28.528       27.978       28.528       490859       0
            2005-02-28       28.379       28.666       28.186       28.264       332954       0
            2005-03-01       28.414       28.637       28.192       28.254       403173       0
            2005-03-02       28.296       28.296       27.844       27.928       450089       0

        """

        for i, test_case in test_cases.items():
            sample = dataset[i]

            assert np.prod(sample['inputs'] == test_case['inputs']), '%d: %s' % (i, 'inputs')
            assert np.prod(sample['outputs'] == test_case['output']), '%d: %s' % (i, 'outputs')
            assert sample['DateTuple'] == test_case['date'], '%d: %s' % (i, 'outputs')
            assert sample['stock'].stock_tckt == test_case['stock'], '%d: %s' % (i, 'outputs')

    def test_batch_generator(self):
        dataset = get_test_dataset()

        batch_size = 5

        def ind_gen(return_as_batch):
            keys = list(self.TEST_CASES)

            x = [keys] * batch_size
            for pair in product(*x):
                if return_as_batch:
                    yield pair
                else:
                    for i in pair:
                        yield i

        gen = batch_generator(dataset, batch_size=batch_size, randomize=False, ind_gen=ind_gen(False))

        for batch_indices, (batch_inputs, batch_outputs) in zip(ind_gen(True), gen):
            assert isinstance(batch_inputs, np.ndarray)
            assert isinstance(batch_outputs, np.ndarray)
            assert batch_outputs.shape[0] == batch_inputs.shape[0] == batch_size
            assert batch_inputs.shape[1] == dataset.time_sample_length
            assert batch_outputs.shape[1] == 2  # "rise" output default is size 2

            inputs = np.stack([self.TEST_CASES[i]['inputs'] for i in batch_indices], axis=0)
            outputs = np.stack([self.TEST_CASES[i]['output'] for i in batch_indices], axis=0)

            assert np.prod(batch_inputs == inputs), '%d: %s' % (batch_indices, 'inputs')
            assert np.prod(batch_outputs == outputs), '%d: %s' % (batch_indices, 'outputs')

    def test_batch_generator_random(self):
        dataset = get_test_dataset()

        batch_size = 1

        last_sample = None
        is_random = False
        for i in range(3):
            gen = batch_generator(dataset, batch_size=batch_size)

            sample = next(gen)
            if last_sample is not None:
                if not all((np.prod(ls == s)) for ls, s in zip(last_sample, sample)):
                    is_random = True
                    break

            last_sample = sample

        assert last_sample is not None
        assert is_random

    def test_normalization(self):
        test_cases = {}
        for ind, test_case in self.TEST_CASES.items():
            new_case = deepcopy(test_case)
            new_case['inputs'] += 1
            test_cases[ind] = new_case

        self._test_samples(test_cases, {'normalization_type': 'debug'})

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





