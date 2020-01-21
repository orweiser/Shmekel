from ..models import get as get_model
from ..datasets import get as get_dataset
from ..losses import get as get_loss
from ..metrics import get as get_metrics
from .results import Results
from .trainer import Trainer
from .backup_handler import get_handler
from Utils.logger import logger
import json


class Experiment:
    """

    """
    def __init__(self, name='default_exp',
                 model_config=None, loss_config=None,
                 train_dataset_config=None, val_dataset_config=None,
                 train_config=None, backup_config=None, metrics_list=None):

        self.model_config = model_config or {'model': 'LSTM'}
        self.loss_config = loss_config or {'loss': 'categorical_crossentropy'}
        self.train_dataset_config = train_dataset_config or {'dataset': 'StocksDataset', 'val_mode': False}
        self.val_dataset_config = val_dataset_config or {'dataset': 'StocksDataset', 'val_mode': True}
        self.train_config = train_config or {}
        self.backup_config = backup_config or dict(project='default_project', handler='DefaultLocal')
        self.metrics_list = metrics_list or ['acc']

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
        self._metrics = None

        # properties declarations
        self._history = None
        self._backup_dir = None
        self._callbacks = None

        # self._create()

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
        self._create()
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
        logger.info('\n' + prefix + ' experiment: ' + str(self) + '\n')
        self._create()
        self.trainer.fit()

    def run(self, backup=True):
        logger.info('running %s', str(self))
        if backup:
            self.backup_handler.dump_config(self.config)

        status = self.status
        if status is 'done':
            logger.info('Experiment is done.')

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
            logger.warning('unexpected status:', status)
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

    def _assert_shapes(self):
        def _get_mode_shapes(mode):
            dataset = getattr(self, '%s_dataset' % mode)
            augmentation = self.trainer.augmentations[mode]

            batch_axis = (10, )

            shapes = []
            shapes.append((batch_axis + tuple(dataset.input_shape),
                           batch_axis + tuple(dataset.output_shape)))
            if augmentation:
                shapes.append(augmentation.get_output_shapes(*shapes[-1]))

            shapes = [(s1[1:], s2[1:]) for s1, s2 in shapes]
            shapes.append((tuple(self.model._input_shape), tuple(self.model._output_shape)))
            # todo: put _shapes in model
            return shapes

        train = _get_mode_shapes('train')
        val = _get_mode_shapes('val')

        for s1, s2 in zip(train, val):
            assert s1 == s2, 'shapes in train and val mode dont match'
            assert len(s1) == 2

        assert train[-1] == train[-2], 'model input and output shape dont match dataset + augmentations' \
                                       '\nGot %s and %s' % (str(train[-1]), str(train[-2]))

    """ Sub-Modules: """
    @property
    def model(self):
        if not self._model:
            self.model_config["input_shape"] = self.train_dataset.input_shape
            self.model_config["output_shape"] = self.train_dataset.output_shape
            self._model = get_model(**self.model_config)
            self.model_config = self._model.config

        return self._model

    @property
    def trainer(self):
        if self._trainer is None:
            self._trainer = Trainer(experiment=self, **self.train_config)
            self.train_config = self._trainer.config
        return self._trainer

    @property
    def loss(self):
        if not self._loss:
            self._loss = get_loss(experiment=self, **self.loss_config)
            self.loss_config = self._loss.config

        return self._loss

    # todo: add support for multiple datasets

    @property
    def train_dataset(self):
        if not self._train_dataset:
            self._train_dataset = get_dataset(**self.train_dataset_config)
            self.train_dataset_config = self._train_dataset.config

        return self._train_dataset

    @property
    def val_dataset(self):
        if not self._val_dataset:
            self._val_dataset = get_dataset(**self.val_dataset_config)
            self.val_dataset_config = self._val_dataset.config

        return self._val_dataset

    @property
    def results(self):
        return Results(experiment=self)

    @property
    def backup_handler(self):
        if self._backup_handler is None:
            self._backup_handler = get_handler(experiment=self, **self.backup_config)
            self.backup_config = self._backup_handler.config

        return self._backup_handler

    """ Configurations: """

    def _create(self):
        _ = self.trainer
        _ = self.loss
        _ = self.train_dataset
        _ = self.val_dataset
        _ = self.model
        _ = self.backup_handler
        _ = self.metrics

    @property
    def config(self):
        self._create()
        return dict(name=self.name,
                    model_config=self.model_config,
                    loss_config=self.loss_config,
                    train_dataset_config=self.train_dataset_config,
                    val_dataset_config=self.val_dataset_config,
                    train_config=self.train_config,
                    backup_config=self.backup_config,
                    metrics_list=self.metrics_list)

    """ Extra properties: """

    @property
    def metrics(self):
        if not self._metrics:
            self._metrics = get_metrics(self.metrics_list)

        return self._metrics

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

    def export(self, epoch, path):  # todo: add defaults
        export_config = {
            'model': self.model_config,
            'weights_path': self.backup_handler.get_snapshot_path(epoch),
            'dataset': {
                'time_sample_length': self.val_dataset.time_sample_length,
                'input_features': self.val_dataset.input_features,
                'output_features': self.val_dataset.output_features,
            }
            # todo: add val_augmentations
        }

        with open(path, 'w') as f:
            json.dump(export_config, f, indent=4)
