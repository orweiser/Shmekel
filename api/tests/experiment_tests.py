from api.tests.base_class import Test
from api.core import Experiment
import numpy as np
from api.tests.dataset_tests import _get_dataset_params
from api.utils.callbacks import DebugCallback
from api.core.backup_handler import BaseBackupHandler
from json_maker import _create_config


class TestExperiment(Test):

    def __init__(self):
        pass

    def get_experiment(self, batch_size=1, epochs=1, add_augmentations=False,
                       time_sample_length=2, add_callback=False, model_config=None) -> Experiment:

        augmentatins_config = [('Debug', {})] if add_augmentations else None
        callback_config = [
            'DebugCallback', ['DebugCallback', {}], {'name': 'DebugCallback'}
        ] if add_callback else None

        params = dict(
            name='test_experiment',
            model_config={
                'input_shape': (time_sample_length, 5), 'model': 'LSTM',
            } if model_config is None else model_config,
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

    def test_model(self):

        model_config = {
            "model": "General_RNN",
            "num_of_layers": 5,
            "layers": [
                {
                    "type": "Dense",
                    "size": 128,
                    "activation_function": "relu",
                    "name": "Layer_0_Dense_128_relu"
                },
                {
                    "type": "KerasLSTM",
                    "size": 8,
                    "name": "Layer_1_KerasLSTM_8_relu",
                    "activation_function": "relu",
                },
                {
                    "type": "KerasLSTM",
                    "size": 128,
                    "name": "Layer_2_KerasLSTM_128_relu",
                    "activation_function": "relu",
                },
                {
                    "type": "Dense",
                    "size": 128,
                    "activation_function": "relu",
                    "name": "Layer_3_Dense_128_relu"
                },
                {
                    "type": "Dense",
                    "size": 32,
                    "activation_function": "relu",
                    "name": "Layer_4_Dense_32_relu"
                }
            ],
            "num_of_rnn_layers": 2,
            "output_activation": "softmax",
            "callbacks": "early_stop"
        }
        exp = self.get_experiment(model_config=model_config)

        exp.model.summary()
        assert exp.model.num_of_layers == model_config['num_of_layers']
        assert exp.model.num_of_rnn_layers == model_config['num_of_rnn_layers']
        assert exp.model.output_activation == model_config['output_activation']
        expected_attributes = {
            'Layer_0_Dense_128_relu': {
                'units': 128,
                'input_shape': (None, 2, 5),
                'output_shape': (None, 2, 128),
            },
            'Layer_1_KerasLSTM_8_relu': {
                'units': 8,
                'input_shape': (None, 2, 128),
                'output_shape': (None, 2, 8),
                'return_sequences': True

            },
            'Layer_2_KerasLSTM_128_relu': {
                'units': 128,
                'input_shape': (None, 2, 8),
                'output_shape': (None, 128),
                'return_sequences': False
            },
            'Layer_3_Dense_128_relu': {
                'units': 128,
                'input_shape': (None, 128),
                'output_shape': (None, 128),
            },
            'Layer_4_Dense_32_relu': {
                'units': 32,
                'input_shape': (None, 128),
                'output_shape': (None, 32),
            }
        }
        for layer_name, attributes in expected_attributes.items():
            layer = exp.model.get_layer(name=layer_name)
            for attribute_name, value in attributes.items():
                assert getattr(layer, attribute_name) == value

    def test_json_maker_create_config(self):
        layer_def = [
            ['Dense', 16, 'relu'],
            ['KerasLSTM', 64, 'sigmoid'],
            ['Dense', 16, 'relu'],
            ['KerasLSTM', 32, 'tanh']
        ]
        config = _create_config(output_activation='debug_activation', layers_def=layer_def)

        gt = {
            "model": "General_RNN",
            "num_of_layers": 4,
            "layers": [
                {
                    "type": "Dense",
                    "size": 16,
                    "activation_function": "relu",
                    "name": "l0_Dense_16_relu"
                },
                {
                    "type": "KerasLSTM",
                    "size": 64,
                    "activation_function": "sigmoid",
                    "name": "l1_KerasLSTM_64_sigmoid"
                },
                {
                    "type": "Dense",
                    "size": 16,
                    "activation_function": "relu",
                    "name": "l2_Dense_16_relu"
                },
                {
                    "type": "KerasLSTM",
                    "size": 32,
                    "activation_function": "tanh",
                    "name": "l3_KerasLSTM_32_tanh"
                }
            ],
            "num_of_rnn_layers": 2,
            "output_activation": "debug_activation",
            "experiment_name": "l0_Dense_16_relu--l1_KerasLSTM_64_sigmoid--l2_Dense_16_relu--l3_KerasLSTM_32_tanh",
            "callbacks": "early_stop"
        }

        assert config == gt
