from api.core import Experiment
import numpy as np
from api.datasets import StocksDataset
from itertools import product
from time import sleep
from copy import deepcopy
from api.tests.dataset_tests import _get_dataset_params, ORIGINAL_TEST_CASES
from api.utils.callbacks import DebugCallback
from api.core.backup_handler import BaseBackupHandler


class TestExperiment:

    def __init__(self):
        pass

    def get_experiment(self, batch_size=1, epochs=1, add_augmentations=False,
                       time_sample_length=2, add_callback=False) -> Experiment:

        augmentatins_config = [('Debug', {})] if add_augmentations else None
        callback_config = [
            'DebugCallback', ['DebugCallback', {}], {'name': 'DebugCallback'}
        ] if add_callback else None

        params = dict(
            name='test_experiment',
            model_config={
                'input_shape': (time_sample_length, 5), 'model': 'LSTM',
            },
            loss_config={'loss': 'categorical_crossentropy'},
            train_dataset_config=_get_dataset_params(time_sample_length=time_sample_length),
            val_dataset_config=_get_dataset_params(time_sample_length=time_sample_length),
            train_config={'batch_size': batch_size, 'epochs': epochs, 'callbacks': callback_config,
                          'train_augmentations': augmentatins_config,
                          'val_augmentations': augmentatins_config},
            backup_config={'handler': 'NullHandler'}, metrics_list=None
        )

        params['train_dataset_config'].update({'val_mode': False})
        params['val_dataset_config'].update({'val_mode': True})

        exp = Experiment(**params)
        return exp

    def test_augmentations(self):
        time_sample_length = 2
        epsilon = 1e-8
        for batch_size in [1, 100]:
            exp = self.get_experiment(add_augmentations=True,
                                      batch_size=batch_size,
                                      time_sample_length=time_sample_length)

            for gen in [exp.trainer.get_batch_generator(mode) for mode in ['train', 'val']]:
                for i, (inputs, outputs) in enumerate(gen):
                    if i > 1000:
                        break

                    assert inputs.shape == (2 * batch_size, time_sample_length, 5)
                    assert outputs.shape == (2 * batch_size, 2)

                    assert np.abs(outputs[:batch_size] == outputs[batch_size:]).max() < epsilon
                    assert np.abs(inputs[:batch_size] - inputs[batch_size:] + 1).max() < epsilon

    def test_callbacks(self):
        exp = self.get_experiment(add_callback=True)

        assert len(exp.callbacks) == 1
        assert isinstance(exp.callbacks[0], BaseBackupHandler)
        assert len(exp.trainer.callbacks) == 4
        assert all((isinstance(c, DebugCallback) for c in exp.trainer.callbacks[1:]))

        exp.start()

        bools = list(c.was_called for c in exp.trainer.callbacks[1:])
        assert all(bools), str(bools)

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





