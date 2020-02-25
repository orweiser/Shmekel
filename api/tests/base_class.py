from api.core import Experiment
import numpy as np
from api.datasets import StocksDataset
from itertools import product
from time import sleep


class Test:

    def run_all(self, verbose=True):
        logs = []
        for k in dir(self):
            if k.startswith('test'):
                try:
                    getattr(self, k)()
                except Exception as e:
                    logs.append('*** FAILED: {0} \t\t {1}'.format(k, getattr(e, 'message', '')))
                else:
                    logs.append('\tPASSED: %s' % k)
        sleep(1)

        if verbose:
            print(*logs, sep='\n')

        return logs




