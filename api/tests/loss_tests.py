from api.losses.classification_reinforce import ClassificationReinforce, ClassificationReinforceMetrics
from keras import Model, Input
import keras.backend as K
import numpy as np
from time import sleep


class TestExperiment:

    def __init__(self):
        pass

    def test_loss_function(self):
        for test in self.cases:
            print(test['loss_params'])
            self._assert_case(batch=test['batch'], loss=ClassificationReinforce('loss', **test['loss_params']))

    cases = [
        {
            'loss_params': dict(win_reward=1, lose_reward=0, additional_rewards=None, mode='linear'),
            'batch': (
                np.array([[1, 0],       [1, 0],     [0, 1],     [1, 0]]),
                np.array([[0.8, 0.2],   [0.1, 0.9], [0.1, 0.9], [0.1, 0.9]]),
                -np.array([0.8, 0.1, 0.9, 0.1]),
            )
        },         {
            'loss_params': dict(win_reward=1, lose_reward=0, additional_rewards={'-1': 0}, mode='linear'),
            'batch': (
                np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]]),
                np.array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7], [0.1, 0.7, 0.2], [0.1, 0.9, 0]]),
                -np.array([0.8, 0.1, 0.7, 0.1]),
            )
        }, {
            'loss_params': dict(win_reward=1, lose_reward=-1, additional_rewards={'-1': 0}, mode='linear'),
            'batch': (
                np.array([[1, 0, 0],         [1, 0, 0],          [0, 1, 0],          [1, 0, 0]]),
                np.array([[0.8, 0.1, 0.1],   [0.1, 0.2, 0.7],    [0.1, 0.7, 0.2],    [0.1, 0.9, 0]]),
                -np.array([0.7,                -0.1,               0.6,                -0.8]),
            )
        }, {
            'loss_params': dict(win_reward=1, lose_reward=0, additional_rewards={'-1': 1}, mode='linear'),
            'batch': (
                np.array([[1, 0, 0],         [1, 0, 0],          [0, 1, 0],          [1, 0, 0]]),
                np.array([[0.8, 0.1, 0.1],   [0.1, 0.2, 0.7],    [0.1, 0.7, 0.2],    [0.1, 0.9, 0]]),
                -np.array([0.9,                0.8,                0.9,                0.1]),
            )
        }, {
            'loss_params': dict(win_reward=1, lose_reward=0, additional_rewards={'-1': 0}, mode='log'),
            'batch': (
                np.array([[1, 0, 0],         [1, 0, 0],          [0, 1, 0],          [1, 0, 0]]),
                np.array([[0.8, 0.1, 0.1],   [0.1, 0.2, 0.7],    [0.1, 0.7, 0.2],    [0.1, 0.9, 0]]),
                -np.log(np.array([0.8,                0.1,                0.7,                0.1])),
            )
        }
    ]
    
    @staticmethod
    def _assert_case(batch, loss: ClassificationReinforce):
        def f(y_true, y_pred):
            _y_true = K.variable(y_true)
            _y_pred = K.variable(y_pred)

            output_tensor = loss.loss(_y_true, _y_pred)
            # K.set_value(_y_true, y_true)
            # K.set_value(_y_pred, y_pred)

            outs = K.eval(output_tensor)
            K.clear_session()
            return outs

        y_true, y_pred, outputs = batch
        outs = f(y_true, y_pred)
        assert np.all((outs - outputs) < 1e-6), '%s, %s' % (str(outs), str(outputs))

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
        sleep(1)
        print(*logs, sep='\n')




