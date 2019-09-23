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

    is_computing_validation_metrics = True

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

    @property
    def callbacks(self):
        return [ClassificationReinforceMetrics(self.experiment)]


class ClassificationReinforceMetrics(Callback):
    def __init__(self, experiment):
        """
        :type experiment: api.core.Experiment
        """
        super(ClassificationReinforceMetrics, self).__init__()
        self.experiment = experiment

    def on_epoch_end(self, epoch, logs=None):
        """

        :param epoch:
        :param logs:
        :return:
        """

        """ 0. sanity - the predict function from model is None """
        assert self.experiment.model.predict_function is None

        """ 1. get validation dataset """
        val_gen = self.experiment.trainer.val_gen
        steps = self.experiment.trainer.validation_steps

        """ 2. predict on val """
        y_true_list = []
        y_pred_list = []

        for i, (x, y) in enumerate(val_gen):
            pred = self.predict(x)

            y_true_list.append(y)
            y_pred_list.append(pred)

            if i + 1 >= steps:
                break

        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)
        del y_true_list, y_pred_list

        """ 3. compute metrics """
        metrics_dict = self.compute_metrics(y_true, y_pred)

        """ 4. log the metrics to history somehow """
        self.log_metrics(metrics_dict)

        """ 5. erase the predict function from model """
        self.experiment.model.predict_function = None

        """ 6. print metrics """
        self.print_metrics(metrics_dict)

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):

        metrics = {}

        uncertain_pred_indices = y_pred.argmax(axis=-1) == y_pred.shape[-1] - 1
        certain_pred_indices = np.logical_not(uncertain_pred_indices)

        metrics['sharpness'] = y_pred.max(axis=-1)
        metrics['acc'] = y_true.argmax(axis=-1) == y_pred.argmax(axis=-1)
        
        if not any(uncertain_pred_indices):
            # always certain

            metrics['certainty_sharpness'] = metrics['sharpness']
            metrics['certain_predictions_acc'] = metrics['acc']
            metrics['uncertainty_sharpness'] = 0
            metrics['uncertain_fraction'] = 0
            metrics['uncertain_2nd_acc'] = 0

        else:
            metrics['uncertain_2nd_acc'] = \
                y_pred[uncertain_pred_indices][:, :-1].argmax(axis=-1) == y_true[uncertain_pred_indices].argmax(axis=-1)

            if not any(certain_pred_indices):
                # always uncertain

                metrics['certainty_sharpness'] = 0
                metrics['certain_predictions_acc'] = 0
                metrics['uncertainty_sharpness'] = metrics['sharpness']
                metrics['uncertain_fraction'] = 1

            else:
                metrics['certainty_sharpness'] = metrics['sharpness'][certain_pred_indices]
                metrics['uncertainty_sharpness'] = metrics['sharpness'][uncertain_pred_indices]
                metrics['certain_predictions_acc'] = metrics['acc'][certain_pred_indices]
                metrics['uncertain_fraction'] = uncertain_pred_indices

        return {'val_' + metric_name: np.mean(value) for metric_name, value in metrics.items()}

    def log_metrics(self, metrics: dict):
        history_callback = self.experiment.model.history

        for k, v in metrics.items():
            history_callback.history.setdefault(k, []).append(v)

    def predict(self, inputs):
        self.experiment.model._make_predict_function()
        predict_function = self.experiment.model.predict_function

        return predict_function([inputs])[0]

    @staticmethod
    def print_metrics(metrics: dict):
        print(' ')
        for key, val in metrics.items():
            print(('%s: ' % key) + ('.' * (30 - len(key))) + ('%0.3f' % float(val)))
        print(' ')
