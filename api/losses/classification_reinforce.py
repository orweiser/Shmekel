from ..core.loss import Loss
import keras.backend as K
from keras.callbacks import Callback
import numpy as np


class ClassificationReinforce(Loss):
    as_tensors: bool
    additional_rewards: dict
    win_reward: int
    lose_reward: int
    mode: str

    def init(self, win_reward=1, lose_reward=0, additional_rewards=None, mode='log', as_tensors=False):
        assert not as_tensors, '"as_tensors" option is not yet supported'
        assert mode in ('linear', 'log')

        self.as_tensors = as_tensors
        self.additional_rewards = additional_rewards or {}
        self.win_reward = win_reward
        self.lose_reward = lose_reward
        self.mode = mode

    def get_default_config(self) -> dict:
        return dict(win_reward=1, lose_reward=0, additional_rewards=None, mode='log', as_tensors=False)

    def __str__(self) -> str:
        s = 'classification_reinforce-'
        for name, key in [('mode',) * 2, ('win', 'win_reward'), ('lose', 'lose_reward')]:
            s += key + '_' + str(self.config[key]) + '-'
        for key, val in self.additional_rewards.items():
            s += str(key) + '_' + str(val) + '-'
        return s[:-1]

    def loss(self, y_true, y_pred):
        epsilon = K.epsilon()

        win_prob = K.sum(y_true * y_pred, axis=-1)
        additional_probs = {ind: y_pred[:, int(ind)] for ind in self.additional_rewards.keys()}

        lose_prob = 1 - (win_prob + sum([v for v in additional_probs.values()]))

        if self.mode == 'log':
            act = lambda prob: K.log(K.clip(prob, min_value=epsilon, max_value=1 - epsilon))
        else:
            act = lambda prob: prob

        reward = float(self.win_reward) * act(win_prob) + \
                 float(self.lose_reward) * act(lose_prob) + \
                 sum([float(additional_reward) * act(additional_probs[str(ind)]) for ind, additional_reward in self.additional_rewards.items()])

        reward = -reward
        return reward


class ClassificationReinforceMetrics(Callback):
    def __init__(self, experiment):
        super(ClassificationReinforceMetrics, self).__init__()
        self.experiment = experiment

    def on_epoch_end(self, epoch, logs=None):
        # 1. get validation dataset
        # 2. predict on val
        # 3. compute metrics
        # 4. log the metrics to history somehow
        pass

    def compute_metrics(self, y_true, y_pred):
        """
        y shape: N [num samples] X 3 [rise, fall, don't know]

        :type y_true: np.ndarray
        :type y_pred: np.ndarray
        :rtype: dict {
                        metric: float
                    }
        """
        raise NotImplementedError()
