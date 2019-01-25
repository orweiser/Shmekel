from .results import Results
from .loss import Loss
from .data import Data
from .trainer import Trainer
from keras import Model


class Experiment:
    def __init__(self, model_config=None, loss_config=None, data_config=None, train_config=None, name=None):
        self.model_config = model_config or {}
        self.loss_config = loss_config or {}
        self.data_config = data_config or {}
        self.train_config = train_config or {}

        # property classes declarations
        self._model = None
        self._results = None
        self._data = None
        self._loss = None
        self._trainer = None

        # properties declarations
        self._name = name
        self._history = None
        self._backup_dir = None

    def run(self, **kwargs):
        print('\nStarting experiment:', self.name, '\n')
        self.trainer.fit()
        self._history = self.trainer.history
        self.backup(**kwargs)

    @property
    def name(self):
        if self._name is not None:
            return self._name
        raise NotImplementedError()

    @property
    def status(self):
        raise NotImplementedError()

    @property
    def model(self):
        """
        :rtype: Model
        :return:
        """
        if self._model is not None:
            return self._model
        raise NotImplementedError()

    @property
    def trainer(self):
        """
        :rtype: Trainer
        :return:
        """
        if not self.train_config:
            pass  # todo
        self._trainer = self._trainer or Trainer(experiment=self, **self.train_config)
        return self._trainer

    @property
    def loss(self):
        """
        :rtype: Loss
        :return:
        """
        self._loss = self._loss or Loss(experiment=self, **self.loss_config)
        return self._loss

    @property
    def metrics(self):
        raise NotImplementedError()

    @property
    def data(self):
        """
        :rtype: Data
        :return:
        """
        self._data = self._data or Data(experiment=self, **self.data_config)
        return self._data

    @property
    def history(self):
        if self._history is not None:
            return self._history
        raise NotImplementedError()

    @property
    def results(self):
        """
        :rtype: Results
        :return:
        """
        self._results = self.results or Results(experiment=self, history=self.history)

        return self._results

    @property
    def backup_dir(self):
        if self._backup_dir is not None:
            return self._backup_dir
        raise NotImplementedError()

    def get_exp_callbacks(self):
        """
        :rtype: list
        :return:
        """
        raise NotImplementedError()

    def backup(self, save_weights: bool=False, save_history: bool=True):
        raise NotImplementedError()

    def erase(self):
        raise NotImplementedError()



