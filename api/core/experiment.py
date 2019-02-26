from ..models import get as get_model
from ..datasets import get as get_dataset
from ..losses import get as get_loss
from .trainer import Trainer


class Project:
    def get_exp_by(self): raise NotImplementedError()


class Experiment:
    """

    """
    def __init__(self, model_config=None, loss_config=None, train_dataset_config=None, val_dataset_config=None, train_config=None):
        self.model_config = model_config or {'model': 'FullyConnected'}
        self.loss_config = loss_config or {'loss': 'categorical_crossentropy'}
        self.train_dataset_config = train_dataset_config or {'dataset': 'MNIST', 'val_mode': False}
        self.val_dataset_config = val_dataset_config or {'dataset': 'MNIST', 'val_mode': True}
        self.train_config = train_config or {}

        # property classes declarations
        self._model = None
        self._results = None
        self._train_dataset = None
        self._val_dataset = None
        self._loss = None
        self._trainer = None

        # properties declarations
        self._name = None
        self._history = None
        self._backup_dir = None
        self._callbacks = None

    def __str__(self):
        s = '\n'
        s += 'Experiment name: ' + self.name + '\n' + '-' * 50 + '\n'
        s += 'Model: ' + str(self.model) + '\n'
        s += 'Loss: ' + str(self.loss) + '\n'
        s += 'Train Dataset: ' + str(self.train_dataset) + '\n'
        s += 'Val Dataset: ' + str(self.val_dataset) + '\n'

        s += '\nStatus: NotImplemented\n'
        if self.results:
            s += '\nResults: NotImplemented\n'

        return s

    def run(self, **kwargs):
        print('\nStarting experiment:', self.name, '\n')
        self.trainer.fit()
        self._history = self.trainer.history
        self.backup(**kwargs)

    @property
    def name(self):
        # todo: implement
        return 'exp_try'

    @property
    def status(self):
        raise NotImplementedError()

    @property
    def model(self):
        if not self._model:
            self._model = get_model(**self.model_config)

        return self._model

    @property
    def trainer(self):
        self._trainer = self._trainer or Trainer(experiment=self, **self.train_config)
        return self._trainer

    @property
    def loss(self):
        if not self._loss:
            self._loss = get_loss(**self.loss_config)

        return self._loss

    @property
    def metrics(self) -> list:
        # todo: implement functionality
        return ['acc']

    @property
    def train_dataset(self):
        if not self._train_dataset:
            self._train_dataset = get_dataset(**self.train_dataset_config)

        return self._train_dataset

    @property
    def val_dataset(self):
        if not self._val_dataset:
            self._val_dataset = get_dataset(**self.val_dataset_config)

        return self._val_dataset

    @property
    def history(self) -> dict:
        if self._history is None:
            self._history = self.results._history or self.trainer._history.history
        return self._history

    @property
    def results(self):
        """
        :rtype: Results
        :return:
        """
        return None
        # return Results(experiment=self, history=self._history or self.trainer._history.history)

    @property
    def callbacks(self) -> list:
        if self._callbacks is None:
            self._callbacks = []
            for obj in [self.loss, self.model]:
                self._callbacks.extend(obj.callbacks)
        return self._callbacks

    def fill_configs(self):
        self.train_config = self.trainer.config
        self.loss_config = self.loss.config
        self.train_dataset_config = self.train_dataset.config
        self.val_dataset_config = self.val_dataset.config
        self.model_config = self.model.config

    """ Paths property and methods. might be moved to a Project class in the future """
    @property
    def backup_dir(self):
        if self._backup_dir is not None:
            return self._backup_dir
        raise NotImplementedError()

    def backup(self, save_weights: bool=False, save_history: bool=True):
        self.fill_configs()
        raise NotImplementedError()

    def erase(self):
        raise NotImplementedError()



