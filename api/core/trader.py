import os
from itertools import product
import numpy as np
from Utils.logger import logger


class Trader:
    def __init__(self, exe_path):
        self.exe_path = exe_path
        self.trained_configuration = None

    @property
    def is_trained(self):
        return self.trained_configuration is not None

    def set_trained(self, configuration):
        self.trained_configuration = configuration

    @property
    def configurable_params(self) -> dict:
        """
            this will output a dictionary with keys that are all the flags that
            the exe supports with all the possible values, using the following syntax:

            - flags with finite number of options will be introduced as sets:
                {"finite_flag": set([0, 1])}

            - flags that supports continuous options will be shown as a tuple of limits:
                {"continuous_flag": (min_val, max_val)}

            - flags with no input value, that can either be stated or not:
                {'action_flag': None}

            Example output:
                return {
                        'elaborate': None,
                        'threshold': (0, 1),
                        'number_of_active_trades': set(range(1, 10000))
                    }

            Using the output of this function, we are able to iterate over all possible
            configurations (at least in the finite case) and choose the configuration that
            best fits our predictor

            This method must be implemented by hand for any new trader / strategy.
        """
        raise NotImplementedError

    def single_run(self, csv_input, configuration, redirect_to=None):
        cmd = '%s %s ' % (self.exe_path, csv_input)

        for key, val in configuration:
            if val is None:
                cmd += '-%s ' % key
            else:
                cmd += '-%s %s ' % (key, str(val))

        if redirect_to:
            cmd += '>> %s' % redirect_to

        logger.info('Running', cmd)
        os.system(cmd)

    def iter_configurations(self, configurable_params=None, resolution=None):
        if configurable_params is None:
            configurable_params = self.configurable_params
        if resolution is None:
            resolution = getattr(self, 'resolution')

        return iterate_flags(configurable_params, resolution)


def _parse_item(value_limits, resolution=None):
    if isinstance(value_limits, set):
        opts = value_limits
        # todo: use resolution here to bound the number of possibilities

    elif isinstance(value_limits, tuple):
        assert len(value_limits) == 2
        if value_limits[0] == value_limits[1]:
            opts = {value_limits[0]}
        else:
            assert resolution is not None, 'When iterating over a continuous parameter, resolution must be stated'
            value_limits = [float(v) for v in value_limits]
            opts = [float(v) for v in np.linspace(*value_limits, num=resolution, endpoint=True)]
    else:
        raise TypeError('unsupported type for values, got %s' % str(type(value_limits)))

    return opts


def iterate_flags(inputs_dict: dict, resolution=None):

    parsed_dict = {}

    DROP = object()
    KEEP = object()

    for key, value_limits in inputs_dict.items():
        opts = _parse_item(value_limits if value_limits is not None else {DROP, KEEP}, resolution)
        parsed_dict[key] = opts

    keys = [k for k in parsed_dict.keys()]
    all_opts = [parsed_dict[k] for k in keys]

    for item in product(*all_opts):
        assert len(item) == len(keys)

        out = {}
        for key, val in zip(keys, item):
            if val is DROP:
                continue
            if val is KEEP:
                val = None
            out[key] = val

        yield out


