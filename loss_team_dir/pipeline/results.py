class Results:
    def __init__(self, experiment=None, history: dict=None):
        self.experiment = experiment

        self.history = None
        if any((history, experiment)):
            self.history = history or experiment.history

    def get_single_epoch(self, epoch=-1):
        num_epochs = [len(v) for v in self.history.values()][0]
        if epoch >= num_epochs:
            raise ValueError("max epoch is", num_epochs - 1, "got", epoch)
        if epoch < 0:
            epoch = num_epochs + epoch

        scores = {k: v[epoch] for k, v in self.history.items()}
        return Epoch(epoch_num=epoch, scores=scores)

    def __getitem__(self, item):
        def get(x):
            if type(x) is str:
                return self.history[x]
            if type(x) is int:
                self.get_single_epoch(x)
            if type(x) is slice:
                epoch_list = []
                for i in range(x.start, x.stop, x.step or 1):
                    epoch_list.append(get(i))
                return epoch_list

        if type(item) is not tuple:
            item = (item,)
            return_list = True
        else:
            return_list = False

        out = []
        for i in item:
            if type(i) in [str, int, slice]:
                out.append(get(i))
            else:
                raise ValueError()

        if not return_list:
            out = out[0]

        return out

    def summary(self):
        """print summary"""
        raise NotImplementedError()


class Epoch:
    def __init__(self, epoch_num, scores):
        self.epoch_num = epoch_num
        self.scores = scores

    def print(self):
        """print epoch"""
        pass
