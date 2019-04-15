from ..models import get as get_model
from ..datasets import get as get_dataset
from ..losses import get as get_loss
from .results import Results
from .trainer import Trainer
from .backup_handler import get_handler


class Experiment:
    """

    """
    def __init__(self, name='default_exp',
                 model_config=None, loss_config=None,
                 train_dataset_config=None, val_dataset_config=None,
                 train_config=None, backup_config=None):

        self.model_config = model_config or {'model': 'LSTM'}
        self.loss_config = loss_config or {'loss': 'categorical_crossentropy'}
        self.train_dataset_config = train_dataset_config or {'dataset': 'StocksDataset', 'val_mode': False}
        self.val_dataset_config = val_dataset_config or {'dataset': 'StocksDataset', 'val_mode': True}
        self.train_config = train_config or {}
        self.backup_config = backup_config or dict(project='default_project', handler='DefaultLocal')

        # todo: have 'project' be a field of experiment instead of backup_handler

        # todo: load existing config by experiment name?
        self.__name = name
        self._name = None

        # property classes declarations
        self._model = None
        self._results = None
        self._train_dataset = None
        self._val_dataset = None
        self._loss = None
        self._trainer = None
        self._backup_handler = None

        # properties declarations
        self._history = None
        self._backup_dir = None
        self._callbacks = None

        self._fill_configs()

    """ Summary methods: """

    @property
    def name(self):
        if self._name is None:
            # todo: assert that if there exists an experiment with the same name, than they have the same config
            self._name = self.__name
        return self._name

    def __str__(self):
        return self.name

    @property
    def status(self):
        self._fill_configs()
        if not self.results:
            return 'initialized'
        else:
            num_trained_epochs = self.results.num_epochs
            num_declared_epochs = self.train_config['epochs']

            if num_trained_epochs == num_declared_epochs:
                return 'done'

            elif num_trained_epochs < num_declared_epochs:
                return 'finished_' + str(num_trained_epochs) + '_out_of_' + str(num_declared_epochs) + '_epochs'

            else:
                return 'unexpected_status--more_trained_epochs_than_declared--got_' + \
                       str(num_trained_epochs) + '_and_' + str(num_declared_epochs)

    def print(self):
        s = '\n'
        s += 'Experiment : ' + str(self) + '\n' + '-' * 50 + '\n'
        s += 'Model: ' + str(self.model) + '\n'
        s += 'Loss: ' + str(self.loss) + '\n'
        s += 'Train Dataset: ' + str(self.train_dataset) + '\n'
        s += 'Val Dataset: ' + str(self.val_dataset) + '\n'

        s += '\nStatus: ' + self.status + '\n'

        print(s)

        if self.results:
            print('Results: ')
            self.results.summary()

    """ Controllers: """

    def start(self, prefix='Starting '):
        print('\n' + prefix + ' experiment:', self, '\n')
        self._fill_configs()
        self.trainer.fit()
        print('\nTraining Done.')

    def run(self, backup=True):
        if backup:
            self.backup_handler.dump_config(self.config)

        status = self.status
        if status is 'done':
            print('Experiment is done.')

        elif status is 'initialized':
            self.start()

        elif status.startswith('finished'):
            snap_i = self.backup_handler.last_snapshot_epoch()

            self.train_config.setdefault('initial_epoch', snap_i)
            _ = self.history
            self._trainer = None
            self.backup_handler.load_snapshot(self.model, snap_i)

            self.start(prefix='Resuming')
            self._history = self.backup_handler.train_logs
            _ = self.train_config.pop('initial_epoch')
            self._trainer = None

        else:
            print('unexpected status:', status)
            return

        self.results.summary()
        if backup:
            self.backup()

    def erase(self, force=False):
        if not force:
            v = input(str(self) +
                      '\nAre you sure you want to erase the entire backup of the experiment? [y,n]')

            if v != 'y':
                print('Aborted.')
                return
        self.backup_handler.erase()

    """ Sub-Modules: """
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

    # todo: add support for multiple datasets

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
    def results(self):
        return Results(experiment=self)

    @property
    def backup_handler(self):
        if self._backup_handler is None:
            self._backup_handler = get_handler(experiment=self, **self.backup_config)
        return self._backup_handler

    """ Configurations: """

    def _fill_configs(self):
        self.train_config = self.trainer.config
        self.loss_config = self.loss.config
        self.train_dataset_config = self.train_dataset.config
        self.val_dataset_config = self.val_dataset.config
        self.model_config = self.model.config
        self.backup_config = self.backup_handler.config

    @property
    def config(self):
        self._fill_configs()
        return dict(name=self.name,
                    model_config=self.model_config,
                    loss_config=self.loss_config,
                    train_dataset_config=self.train_dataset_config,
                    val_dataset_config=self.val_dataset_config,
                    train_config=self.train_config,
                    backup_config=self.backup_config)

    """ Extra properties: """

    @property
    def metrics(self) -> list:
        # todo: implement functionality
        return ['acc']

    @property
    def history(self) -> dict:
        if self._history is None:
            if self.trainer._history:
                self._history = self.trainer._history.history
            else:
                i = self.backup_handler.last_history_epoch()
                if i:
                    self._history = self.backup_handler.load_history(i)

        return self._history

    @property
    def callbacks(self) -> list:
        if self._callbacks is None:
            self._callbacks = []
            for obj in [self.loss, self.model, self.train_dataset, self.val_dataset]:
                if hasattr(obj, 'callbacks'):
                    self._callbacks.extend(obj.callbacks)

            self._callbacks.append(self.backup_handler)
        return self._callbacks

    """ Paths property and methods. might be moved to a Project class in the future """
    def backup(self):
        self.backup_handler.dump_history(self.history or {}, epoch=-1)
        self.backup_handler.dump_snapshot(self.model, epoch=-1)
        self.backup_handler.dump_config(self.config)

    """ Methods yet to be implemented: """
    def compute_shapes(self):
        """ computes the input and output shapes (considering augmentations) """
        raise NotImplementedError()



