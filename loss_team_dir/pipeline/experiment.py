from .results import Results
from .losses import get as get_loss
from .data import Data
from .trainer import Trainer
from .models.model import Model


class Experiment:
    def __init__(self, model_config=None, loss_config=None, data_config=None,
                 train_config=None, name=None, callback=None):
        self.model_config = model_config or {'model': 'lstm'}
        self.loss_config = loss_config or {'loss': 'sparse_categorical_crossentropy'}
        self.data_config = data_config or {}
        self.train_config = train_config or {}
        self.callback = callback



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
        # self.backup(**kwargs)

    @property
    def name(self):
        if self._name is not None:
            return self._name
        # todo: implement
        return 'exp_try'

    @property
    def status(self):
        raise NotImplementedError()

    @property
    def model(self):
        """
        :rtype: Model
        :return:
        """
        if self._model is None:
            self._model = Model(experiment=self, **self.model_config)

        return self._model

    @property
    def trainer(self):
        """
        :rtype: Trainer
        :return:
        """
        self._trainer = self._trainer or Trainer(experiment=self, **self.train_config)
        return self._trainer

    @property
    def loss(self):
        """
        :rtype: Loss
        :return:
        """
        self._loss = self._loss or get_loss(experiment=self, **self.loss_config)
        return self._loss

    @property
    def metrics(self):
        # todo:
        #  raise NotImplementedError()
        return ['acc']

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
        if self._history is None:
            self._history = self.results._history or self.trainer._history.history
        return self._history

    @property
    def results(self):
        """
        :rtype: Results
        :return:
        """
        return Results(experiment=self, history=self._history or self.trainer._history.history)

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
        callbacks = []
        for obj in [self.loss, self.data, self.model]:
            callbacks.extend(obj.callbacks)
        return callbacks

    def backup(self, save_weights: bool=False, save_history: bool=True):
        self.fill_configs()
        raise NotImplementedError()

    def fill_configs(self):
        self.train_config = self.trainer.config
        self.loss_config = self.loss.config
        self.data_config = self.data.config
        self.model_config = self.model.config

    def erase(self):
        raise NotImplementedError()

    def save_model(self, model_flag=True, weights_flag=True, path_name='', model_name='model'):

        weights_name = model_name + 'weights'
        model_json = self._model.to_json()
        if model_flag:
            with open(model_name + path_name + ".json", "w") as json_file:
                json_file.write(model_json)
        if weights_flag:
            self._model.save_weights(weights_name + ".h5")
        print("Saved model to disk")
