from keras.callbacks import Callback
import os


def get_handler(handler='DefaultLocal', **kwargs):
    if handler == 'NullHandler':
        return NullHandler(handler='NullHandler', **kwargs)

    if handler == 'DefaultLocal':
        return DefaultLocal(handler='DefaultLocal', **kwargs)

    if handler == 'DefaultLossGroup':
        return DefaultLocal(handler='DefaultLossGroup', **kwargs)


class BaseBackupHandler(Callback):
    def __init__(self, experiment, handler='default_local', project='',
                 snapshot_backup_delta=1, history_backup_delta=1,
                 backup_history_after_training=True, backup_weights_after_training=True):
        super(BaseBackupHandler, self).__init__()

        self.experiment = experiment
        self.project = project

        self.snapshot_backup_delta = snapshot_backup_delta
        self.history_backup_delta = history_backup_delta
        self.backup_history_after_training = backup_history_after_training
        self.backup_weights_after_training = backup_weights_after_training

        self.config = dict(project=project, handler=handler,
                           snapshot_backup_delta=snapshot_backup_delta,
                           history_backup_delta=history_backup_delta,
                           backup_weights_after_training=backup_weights_after_training,
                           backup_history_after_training=backup_history_after_training)

    """ Base Paths: """
    @property
    def res_dir_absolute_path(self):
        raise NotImplementedError()

    @property
    def exp_absolute_path(self):
        return os.path.join(self.res_dir_absolute_path, self.project, str(self.experiment))

    """ Handle Config: """

    def get_config_path(self):
        return os.path.join(self.exp_absolute_path, "config.json")

    def dump_config(self):
        raise NotImplementedError()

    @staticmethod
    def load_config(path):
        raise NotImplementedError()

    """ Handle Snapshots: """
    @property
    def snapshots_dir_relative_path(self):
        return 'snapshots'

    def get_snapshot_path(self, epoch: int):
        if epoch < 0:
            epoch = self.experiment.train_config['epoch'] + epoch + 1

        return os.path.join(self.exp_absolute_path, self.snapshots_dir_relative_path,
                            "snapshot_EPOCH.h5".replace('EPOCH', str(epoch)))

    def dump_snapshot(self, model, epoch: int):
        raise NotImplementedError()

    def load_snapshot(self, model, epoch: int):
        raise NotImplementedError()

    """ Handle history files: """
    @property
    def histories_dir_relative_path(self):
        return "histories"

    def get_history_path(self, epoch: int):
        if epoch < 0:
            epoch = self.experiment.train_config['epochs'] + epoch + 1

        return os.path.join(self.exp_absolute_path, self.histories_dir_relative_path,
                            "history_EPOCH.h5".replace('EPOCH', str(epoch)))

    def dump_history(self, history: dict, epoch: int):
        raise NotImplementedError()

    def load_history(self, epoch: int):
        raise NotImplementedError()

    """ Callback methods: """
    def on_train_begin(self, logs=None):
        self.train_logs = {}

    def on_epoch_end(self, epoch, logs=None):
        epoch = epoch + 1
        backup_last = epoch == self.experiment.train_config['epochs']

        for key, item in logs.items():
            self.train_logs.setdefault(key, []).append(item)

        if (self.history_backup_delta and not epoch % self.history_backup_delta) or \
                (backup_last and self.backup_history_after_training):
            self.dump_history(history=self.train_logs, epoch=epoch)

        if (self.snapshot_backup_delta and not epoch % self.snapshot_backup_delta) or \
                (backup_last and self.backup_weights_after_training):
            self.dump_snapshot(self.experiment.model, epoch=epoch)

    """ More: """
    def erase(self):
        raise NotImplementedError()


class NullHandler(BaseBackupHandler):

    def dump_config(self):
        pass

    @staticmethod
    def load_config(path):
        pass

    @property
    def res_dir_absolute_path(self):
        return ''

    def dump_snapshot(self, model, epoch: int):
        pass

    def load_snapshot(self, model, epoch: int):
        pass

    def dump_history(self, history: dict, epoch: int):
        pass

    def load_history(self, epoch: int):
        pass

    def erase(self):
        pass


class DefaultLocal(BaseBackupHandler):
    from shutil import rmtree
    import pickle

    def __init__(self, **kwargs):
        super(DefaultLocal, self).__init__(**kwargs)

    @property
    def res_dir_absolute_path(self):
        return os.path.abspath(os.path.join(os.path.pardir, self.project, 'Shmekel_Results'))

    def dump_snapshot(self, model, epoch: int):
        if not os.path.exists(os.path.join(self.exp_absolute_path, self.snapshots_dir_relative_path)):
            os.makedirs(os.path.join(self.exp_absolute_path, self.snapshots_dir_relative_path))

        model.save_weights(self.get_snapshot_path(epoch))

    def load_snapshot(self, model, epoch: int):
        if not os.path.exists(self.get_snapshot_path(epoch)):
            return None

        model.load_weights(self.get_snapshot_path(epoch))

    def dump_history(self, history: dict, epoch: int):
        if not os.path.exists(os.path.join(self.exp_absolute_path, self.histories_dir_relative_path)):
            os.makedirs(os.path.join(self.exp_absolute_path, self.histories_dir_relative_path))
        with open(self.get_history_path(epoch), 'wb') as f:
            self.pickle.dump(history, f)

    def load_history(self, epoch: int):
        if not os.path.exists(self.get_history_path(epoch)):
            return None

        with open(self.get_history_path(epoch), 'rb') as f:
            h = self.pickle.load(f)

        return h

    def erase(self):
        self.rmtree(self.exp_absolute_path)


class DefaultLossGroup(BaseBackupHandler):
    from shutil import rmtree
    import pickle

    def __init__(self, **kwargs):
        super(DefaultLossGroup, self).__init__(**kwargs)

    @property
    def res_dir_absolute_path(self):
        return "C:\\Users\Danielle\Google Drive\Shmekels_drive\shmekel_loss_results"

    def get_history_path(self, epoch: int):
        return os.path.join(self.experiment, 'history.h5')

    def dump_snapshot(self, model, epoch: int):
        model.save_weights(self.get_snapshot_path(epoch))

    def load_snapshot(self, model, epoch: int):
        model.load_weights(self.get_snapshot_path(epoch))

    def dump_history(self, history: dict, epoch: int):
        if not os.path.exists(os.path.join(self.exp_absolute_path, self.histories_dir_relative_path)):
            os.makedirs(os.path.join(self.exp_absolute_path, self.histories_dir_relative_path))
        with open(self.get_history_path(epoch), 'wb') as f:
            self.pickle.dump(history, f)

    def load_history(self, epoch: int):
        with open(self.get_history_path(epoch), 'rb') as f:
            h = self.pickle.load(f)

        return h

    def erase(self):
        self.rmtree(self.exp_absolute_path)
