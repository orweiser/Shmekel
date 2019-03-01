import matplotlib.pyplot as plt
import numpy as np


class Results:
    def __init__(self, experiment=None, history: dict=None):
        assert any((experiment, history))
        self.experiment = experiment

        self._history = history
        self._epoch_list = []

    """ Analysis tools: """

    def plot(self, metrics=None):
        metrics = metrics or self.metrics_list
        assert all([m in self.metrics_list for m in metrics]), 'one or more of the metrics is not in metrics list'

        fig = plt.figure()
        for metric in metrics:
            plt.plot(self[metric])
        plt.legend(metrics)
        plt.show()

    def get_best_epoch(self, metric=None):
        metric = metric or 'val_acc'
        metric = metric or 'acc'
        assert metric and metric in self.metrics_list, 'metric ' + str(metric) + ' not in metrics list'

        metric_history = np.array(self[metric])
        return self[metric_history.argmax()]

    def summary(self):
        print(self)
        if self:
            print('Metrics:', *self.metrics_list)
            print('Number of Epochs:', self.num_epochs)
            print('Best Epoch:\n')
            self.get_best_epoch().print()
        else:
            print('No results found')

    """ Analysis data: """

    @property
    def history(self):
        if self._history is None:
            self._history = self.experiment.backup_handler.load_history(epoch=-1)

        return self._history

    @property
    def epoch_list(self):
        for i in range(len(self._epoch_list), self.num_epochs):
            self._epoch_list.append(self.get_single_epoch(epoch=i))
        return self._epoch_list

    @property
    def num_epochs(self):
        if not self:
            return 0
        return len(self.history[self.metrics_list[0]])

    @property
    def metrics_list(self):
        if not self:
            return None
        return [key for key in self.history.keys()]

    """ Getters: """

    def get_single_epoch(self, epoch=-1):
        num_epochs = self.num_epochs
        if epoch >= num_epochs:
            raise ValueError("max epoch is", num_epochs - 1, "got", epoch)
        if epoch < 0:
            epoch = num_epochs + epoch

        scores = {k: v[epoch] for k, v in self.history.items()}
        return Epoch(epoch_num=epoch, scores=scores, results_parent=self)

    def __getitem__(self, item):
        def get(x):
            if type(x) is str:
                return self.history[x]
            if type(x) is int:
                return self.epoch_list[x]
            if type(x) is slice:
                epoch_list = []
                for i in range(x.run, x.stop, x.step or 1):
                    epoch_list.append(get(i))
                return epoch_list

        if not isinstance(item, (list, tuple)):
            item = (item,)
            return_list = False
        else:
            return_list = True

        out = []
        for i in item:
            if type(i) in [str, int, slice]:
                out.append(get(i))
            else:
                raise TypeError('unexpected item type. got: ' + str(type(i)))

        if not return_list:
            out = out[0]

        return out

    """ Descriptors """

    def __bool__(self):
        return bool(self.history)

    def __str__(self):
        return str(self.experiment) + '--Results'


class Epoch:
    def __init__(self, epoch_num, scores: dict, results_parent: Results):
        self.epoch_num = epoch_num
        self.scores = scores

        self.results_parent = results_parent

    def __getitem__(self, item): return self.scores.__getitem__(item)

    def __iter__(self): return self.scores.__iter__()

    def print(self, indent=''):
        print(indent + 'Epoch', self.epoch_num, '\n' + '-' * 50)
        for m in self.results_parent.metrics_list:
            print(indent + '-->', m, '.' * (30 - len(m)), np.round(self[m], 3))

