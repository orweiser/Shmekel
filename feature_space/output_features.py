import numpy as np
from shmekel_core import Feature
from utils.generic_utils import one_hot


"""
NEGATIVE TIME DELAY
"""


class Rise(Feature):
    """
        indicates a rise in the NEXT day price ('Close' - 'High')

        "output_type": this feature supports various output types:
            - 'regression': return the difference 'Close' - 'High'
            - 'binary':
                    if 'close' > 'high':
                        return 1
                    else:
                        return 0
            - 'ternary':
                if 'Close' - 'High' < "threshold":
                    return 0
                elif 'Close' > 'High':
                    return 1
                else:
                    return -1

            - 'categorical': return the 1-hot representation of "binary" if threshold is not specified,
                    else returns the 1-hot representation of the ternary output type

        """
    SUPPORTED_OUTPUT_TYPES = ('regression', 'binary', 'categorical', 'ternary')

    def __init__(self, output_type='categorical', threshold=None, k_next_candle=1, normalization_type=None, **kwargs):
        assert output_type in self.SUPPORTED_OUTPUT_TYPES
        assert not all((threshold is None, output_type == 'ternary')), 'when using "outpit_type" == "ternary", ' \
                                                                       'must specify threshold'

        super(Rise, self).__init__(normalization_type=normalization_type, **kwargs)
        self.time_delay = -k_next_candle
        self.num_features = 1 if output_type != 'categorical' else (2 if threshold is None else 3)

        self.output_type = output_type
        self.threshold = threshold
        self.k_next_candle = k_next_candle

    def to_ternary(self, diff, threshold=None):
        threshold = threshold if threshold is not None else self.threshold
        assert threshold is not None

        changed_ind = np.abs(diff) > self.threshold

        return 1 * (changed_ind * (diff > 0) - changed_ind * (diff < 0))

    @staticmethod
    def to_binary(diff):
        return 1 * (diff > 0)

    def to_categorical(self, diff):
        if self.threshold is None:
            diff = self.to_binary(diff)
        else:
            diff = self.to_ternary(diff)
            diff += 1

        return self._to_1_hot(diff)

    def _to_1_hot(self, y, num_classes=None):
        num_classes = num_classes or self.num_features
        return one_hot(y, num_classes)

    def _compute_feature(self, data):
        open = self._get_basic_feature(data[0], 'open')
        close = self._get_basic_feature(data[0], 'close')

        diff = close - open

        if self.output_type == 'regression':
            pass

        elif self.output_type == 'binary':
            diff = self.to_binary(diff)

        elif self.output_type == 'ternary':
            diff = self.to_ternary(diff)

        elif self.output_type == 'categorical':
            diff = self.to_categorical(diff)

        else:
            raise RuntimeError('unexpected "output_type": ' + self.output_type)

        return diff[:-self.k_next_candle]
