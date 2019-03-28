import keras.backend as K


class PredictionDependentLoss:
    """
    loss number 3 on the board

    if p is the prediction, y is the true label, ~ is uncertain and l is the categorical loss, than:
    loss = (p != ~) * l(y) + a * (p == ~) * l(y) + b * (p != y) * l(~)
    """
    def __init__(self, weights=(-1, 0), flip_sign=False):
        self.weights = self.__weights(weights)
        self.flip_sign = flip_sign

    def __weights(self, weights):
        if type(weights) is int:
            weights = (weights,)

        if len(weights) == 1:
            return 1, weights[0], 0
        elif len(weights) == 2:
            return 1, weights[0], weights[1]
        else:
            return [w for w in weights]

    def _weighted_losses(self, y_true, y_pred):
        uncertain_ind = K.int_shape(y_pred)[-1] - 1

        uc_bool = K.cast(K.equal(K.argmax(y_pred, axis=-1), uncertain_ind), 'float32')
        c_bool = K.cast(K.not_equal(K.argmax(y_pred, axis=-1), uncertain_ind), 'float32')
        mistakes_bool = K.cast(K.not_equal(K.argmax(y_pred, axis=-1), K.argmax(y_true, axis=-1)), 'float32')

        loss = K.categorical_crossentropy(y_true, y_pred)

        epsilon = K.epsilon()
        alt_loss = -K.log(K.clip(y_pred[:, -1], epsilon, 1. - epsilon))

        weighted_losses = [
            self.weights[0] * (c_bool * loss),
            self.weights[1] * (uc_bool * loss),
            self.weights[2] * (mistakes_bool * alt_loss)
        ]
        return weighted_losses

    def __call__(self, y_true, y_pred):
        weighted_losses = self._weighted_losses(y_true, y_pred)
        return (-1 if self.flip_sign else 1) * sum(weighted_losses)


class __ReinforceLosses:
    """
    A parent class for reinforce losses
    """
    def __init__(self, rewards=(-1, 0), minus=True, without_uncertainty=False):
        self.rewards = self.__rewards(rewards)
        self.minus = minus
        self.without_uncertainty = without_uncertainty

    def __rewards(self, rewards):
        if type(rewards) is int:
            rewards = (rewards,)

        if len(rewards) == 1:
            return 1, rewards[0], 0
        elif len(rewards) == 2:
            if self.without_uncertainty:
                return rewards[0], rewards[1], 0
            return 1, rewards[0], rewards[1]
        else:
            return [reward for reward in rewards]

    def _expected_rewards(self, y_true, y_pred):
        pass

    def __call__(self, y_true, y_pred):
        expected_rewards = self._expected_rewards(y_true, y_pred)
        return (-1 if self.minus else 1) * sum(expected_rewards)


class LinearReinforce(__ReinforceLosses):
    """
    reinforce loss as it is, without applying log() on the predictions.

    if ~ is the uncertainty class, p is the distribution output of the model and y is the true class, than:
    loss = p[y] + a * sum(p[i != y && i != ~]) + b * p[~]
    """
    def __init__(self, *args, **kwargs):
        super(LinearReinforce, self).__init__(*args, **kwargs)

    def _expected_rewards(self, y_true, y_pred):
        correct_prob = K.sum(y_true * y_pred, axis=-1)
        uncertain_prob = y_pred[:, -1] if not self.without_uncertainty else 0
        error_prob = 1 - (correct_prob + uncertain_prob)

        expected_rewards = [
            self.rewards[0] * correct_prob,
            self.rewards[1] * error_prob,
        ]

        if not self.without_uncertainty:
            expected_rewards.append(
                self.rewards[2] * uncertain_prob
            )

        return expected_rewards


class LogReinforce(__ReinforceLosses):
    """
    reinforce loss after applying log() on the predictions.

    if ~ is the uncertainty class, p is the distribution output of the model and y is the true class, than:
    loss = log(p[y]) + a * log(sum(p[i != y && i != ~])) + b * log(p[~])
    """
    def __init__(self, *args, **kwargs):
        super(LogReinforce, self).__init__(*args, **kwargs)

    def _expected_rewards(self, y_true, y_pred):
        epsilon = K.epsilon()

        correct_prob = K.sum(y_true * y_pred, axis=-1)
        uncertain_prob = y_pred[:, -1] if not self.without_uncertainty else 0
        error_prob = 1 - (correct_prob + uncertain_prob)

        expected_rewards = [
            self.rewards[0] * K.log(K.clip(correct_prob, min_value=epsilon, max_value=1 - epsilon)),
            self.rewards[1] * K.log(K.clip(error_prob, min_value=epsilon, max_value=1 - epsilon)),
        ]

        if not self.without_uncertainty:
            expected_rewards.append(
                self.rewards[2] * K.log(K.clip(uncertain_prob, min_value=epsilon, max_value=1 - epsilon))
            )

        return expected_rewards


# class LogLinearReinforce(__ReinforceLosses):
#     def __init__(self, *args, **kwargs):
#         super(LogLinearReinforce, self).__init__(*args, **kwargs)
#
#     def _expected_rewards(self, y_true, y_pred):
#         epsilon = K.epsilon()
#
#         correct_prob = K.sum(y_true * y_pred, axis=-1)
#         uncertain_prob = y_pred[:, -1] if not self.without_uncertainty else 0
#         error_prob = 1 - (correct_prob + uncertain_prob)
#
#         expected_rewards = [
#             self.rewards[0] * correct_prob * K.log(K.clip(correct_prob, min_value=epsilon, max_value=1 - epsilon)),
#             self.rewards[1] * error_prob * K.log(K.clip(error_prob, min_value=epsilon, max_value=1 - epsilon)),
#         ]
#
#         if not self.without_uncertainty:
#             expected_rewards.append(
#                 self.rewards[2] * uncertain_prob * K.log(K.clip(uncertain_prob, min_value=epsilon, max_value=1 - epsilon))
#             )
#
#         return expected_rewards

