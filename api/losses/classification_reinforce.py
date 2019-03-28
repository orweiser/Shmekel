from ..core.loss import Loss
import keras.backend as K


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
        additional_probs = {ind: y_pred[:, ind] for ind in self.additional_rewards.keys()}

        lose_prob = 1 - (win_prob + sum([v for v in additional_probs.values()]))

        if self.mode == 'log':
            act = lambda prob: K.log(K.clip(prob, min_value=epsilon, max_value=1 - epsilon))
        else:
            act = lambda prob: prob

        reward = self.win_reward * act(win_prob) + \
                 self.lose_reward * act(lose_prob) + \
                 sum([reward * act(additional_probs[ind]) for ind, reward in self.additional_rewards.items()])

        reward = -reward
        return reward
