"""
Backup handlers to store and read experiments and handle paths
"""

from keras.callbacks import Callback
from keras.models import load_model
import os
from Utils.logger import logger


def get_handler(handler='DefaultLocal', instantiate=True, **kwargs):
    """
    a handler getter by subclass name

    :type handler: str
    :param handler: the name of the handler to use

    :param instantiate: bool.
            if True:
                passes **kwargs to the subclass and returns an instance
            if False:
                returns the subclass without instantiating (ignores **kwargs)

    :param kwargs: extra parameters to pass to the handler

    :rtype: BaseBackupHandler
    :return: handler instance
    """
    handlers = {
        'NullHandler': NullHandler,
        'DefaultLocal': DefaultLocal,
        'DefaultLossGroup': DefaultLossGroup,
        'DefaultModelGroup': DefaultModelGroup
    }

    try:
        clas = handlers[handler]
    except KeyError:
        clas = globals()[handler]

    if instantiate:
        return clas(**kwargs)
    else:
        return clas


class BaseBackupHandler(Callback):
    def __init__(self, experiment, handler=None,
                 snapshot_backup_delta=1, history_backup_delta=1,
                 save_history_after_training=True, save_snapshot_after_training=True, project=None):
        """
        :type experiment: api.core.Experiment

        :type handler: str
        :param handler: name of handler

        :param project: str. name of the project

        :param snapshot_backup_delta: the number of epochs that pass between saving snapshots
                                to deactivate saving snapshots during train, set it equal to zero.
        :param history_backup_delta: same as snapshot_backup_delta, only for train history files

        :param save_history_after_training: bool. whether or not to save training history at the end of training
        :param save_snapshot_after_training: bool. whether or not to a snapshot at the end of training
        """
        super(BaseBackupHandler, self).__init__()

        if project is not None:
            raise RuntimeError('"project" argument is deprecated - please remove it from the config. '
                               'instead, use project/exp_name '
                               'convention in experiment.name')

        handler = handler or self.__class__.__name__

        self.experiment = experiment

        self.snapshot_backup_delta = snapshot_backup_delta
        self.history_backup_delta = history_backup_delta
        self.save_history_after_training = save_history_after_training
        self.save_snapshot_after_training = save_snapshot_after_training

        self.config = dict(handler=handler,
                           snapshot_backup_delta=snapshot_backup_delta,
                           history_backup_delta=history_backup_delta,
                           save_snapshot_after_training=save_snapshot_after_training,
                           save_history_after_training=save_history_after_training)

    """ Base Paths: """
    @property
    def res_dir_absolute_path(self):
        """
        :return: an absolute path to the results directory, in which
            all the different configs are stored.
        """
        raise NotImplementedError()

    @property
    def exp_absolute_path(self):
        """
        :return: absolute path to the experiment backup directory.
            default: res_dir/experiment
        """
        return os.path.join(self.res_dir_absolute_path, self.experiment.name)

    """ Handle Config: """

    def get_config_path(self):
        """
        :return: a path to experiment config backup path
            default: exp_dir/config.json
        """
        return os.path.join(self.exp_absolute_path, "config.json")

    def dump_config(self, config: dict):
        """
        a method to backup a config file
        """
        raise NotImplementedError()

    @staticmethod
    def load_config(path):
        """
        a complimentary method to "dump_config" that reads from storage.
        this method should be static (doesn't get "self" as an input)
        """
        raise NotImplementedError()

    """ Handle Snapshots: """
    @property
    def snapshots_dir_relative_path(self):
        """ returns the name of the snapshots dir. default: "snapshots" """
        return 'snapshots'

    def get_snapshot_path(self, epoch: int):
        """ get a snapshot path according to epoch_number (counting starts from 1) """
        if epoch < 0:
            epoch = self.experiment.train_config['epochs'] + epoch + 1

        return os.path.join(self.exp_absolute_path, self.snapshots_dir_relative_path,
                            "snapshot_EPOCH.h5".replace('EPOCH', str(epoch)))

    def last_snapshot_epoch(self) -> int:
        """ get the latest epoch number with saved snapshot. """
        for i in range(self.experiment.train_config['epochs'], 0, -1):
            if os.path.exists(self.get_snapshot_path(i)):
                return i

    def dump_snapshot(self, model, epoch: int):
        """ method to implement the saving of snapshots """
        raise NotImplementedError()

    def load_snapshot(self, model, epoch: int):
        """ method to implement the loading of snapshots """
        raise NotImplementedError()

    def load_snapshot_from_path(self, model, path: str):
        """ method to implement the loading of snapshots from a given epoch snapshot path"""
        raise NotImplementedError()

    """ Handle history files: """
    @property
    def histories_dir_relative_path(self):
        """ returns the name of the histories directory. default: "histories" """
        return "histories"

    def get_history_path(self, epoch: int):
        """ get a training history path according to epoch_number (counting starts from 1) """
        if epoch < 0:
            epoch = self.experiment.train_config['epochs'] + epoch + 1

        return os.path.join(self.exp_absolute_path, self.histories_dir_relative_path,
                            "history_EPOCH.h5".replace('EPOCH', str(epoch)))

    def last_history_epoch(self):
        """ get the latest epoch number with saved history. """
        for i in range(self.experiment.train_config['epochs'], 0, -1):
            if os.path.exists(self.get_history_path(i)):
                return i

    def dump_history(self, history: dict, epoch: int):
        """ method to implement the saving of training histories """
        raise NotImplementedError()

    def load_history(self, epoch: int):
        """ method to implement the reading of saved histories """
        raise NotImplementedError()

    """ """
    @property
    def exported_dir_relative_path(self):
        return 'exported_configs'

    def get_exported_path(self, epoch: int):
        """ get a training history path according to epoch_number (counting starts from 1) """
        if epoch < 0:
            epoch = self.experiment.train_config['epochs'] + epoch + 1

        return os.path.join(self.exp_absolute_path, self.exported_dir_relative_path,
                            "exported_EPOCH.json".replace('EPOCH', str(epoch)))

    def dump_exported(self, epoch: int):
        """ method to implement the saving of training histories """
        raise NotImplementedError()

    def load_exported(self, epoch: int):
        """ method to implement the reading of saved histories """
        raise NotImplementedError()

    """ Callback methods: """
    """
            below are methods that implement the saving 
            of snapshots and histories during and 
            after training.
    """
    def on_train_begin(self, logs=None):
        self.train_logs = None
        if 'initial_epoch' in self.experiment.train_config.keys():
            epoch = self.experiment.train_config['initial_epoch']
            if os.path.exists(self.get_history_path(epoch)):
                self.train_logs = self.load_history(epoch)
            # todo: handle other cases

        self.train_logs = self.train_logs or {}

    def on_epoch_end(self, epoch, logs=None):
        epoch = epoch + 1
        backup_last = epoch == self.experiment.train_config['epochs']

        for key, item in logs.items():
            self.train_logs.setdefault(key, []).append(item)

        if (self.history_backup_delta and not epoch % self.history_backup_delta) or \
                (backup_last and self.save_history_after_training):
            self.dump_history(history=self.train_logs, epoch=epoch)

        if (self.snapshot_backup_delta and not epoch % self.snapshot_backup_delta) or \
                (backup_last and self.save_snapshot_after_training):
            self.dump_snapshot(self.experiment.model, epoch=epoch)

        self.dump_exported(epoch)

    """ More: """
    def erase(self):
        """ a method to erase the backup of an experiment. """
        raise NotImplementedError()


class NullHandler(BaseBackupHandler):
    """
        a null backup handler that doesnt backup or read anything,
        to use in experiments you do not want to backup at all
    """

    def dump_exported(self, epoch: int):
        pass

    def load_exported(self, epoch: int):
        pass

    def dump_config(self, config: dict):
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
    """
    saves and reads file locally on computer.

    definitions:
        -- snapshots are saved and loaded via keras
        -- configs are saved as json files
        -- histories are saved as "h5" files
    """

    def dump_exported(self, epoch: int):
        exported_config = self.experiment.export(epoch)
        if not os.path.exists(os.path.join(self.exp_absolute_path, self.exported_dir_relative_path)):
            os.makedirs(os.path.join(self.exp_absolute_path, self.exported_dir_relative_path))
        with open(self.get_exported_path(epoch), 'w') as f:
            self.json.dump(exported_config, f, indent=4)

    def load_exported(self, epoch: int):
        with open(self.get_exported_path(epoch)) as f:
            h = self.json.load(f)

        return h

    import shutil
    import pickle
    import json
    import io

    def __init__(self, **kwargs):
        super(DefaultLocal, self).__init__(**kwargs)

    @property
    def res_dir_absolute_path(self):
        return os.path.abspath(os.path.join(os.path.pardir, 'Shmekel_Results'))

    @logger.debug_dec
    def dump_snapshot(self, model, epoch: int):
        if not os.path.exists(os.path.join(self.exp_absolute_path, self.snapshots_dir_relative_path)):
            os.makedirs(os.path.join(self.exp_absolute_path, self.snapshots_dir_relative_path))

        path = self.get_snapshot_path(epoch)
        logger.debug('saving snapshot epoch %d at %s', epoch, path)
        model.save_weights(path)

    @logger.debug_dec
    def load_snapshot(self, model, epoch: int):
        path = self.get_snapshot_path(epoch)

        if not os.path.exists(path):
            return None

        logger.debug('loading snapshot epoch %d from %s', epoch, path)
        model.load_weights(path)

    def load_snapshot_from_path(self, model, path: str):
        model.load_weights(os.path.join(self.res_dir_absolute_path, path))

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
        path = str(self.exp_absolute_path)
        self.shutil.rmtree(path)

    def dump_config(self, config: dict):
        path = self.get_config_path()
        dir_path = path.rsplit(os.path.sep, 1)[0]

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with self.io.open(path, 'w', encoding='utf8') as outfile:
            outfile.write(self.json.dumps(config, indent=4, sort_keys=True,
                                          separators=(',', ': '), ensure_ascii=False))

    @staticmethod
    def load_config(path):
        import json
        with open(path, 'r') as f:
            data = f.read()
            c = json.loads(data)
        return c


class DefaultLossGroup(BaseBackupHandler):
    """ will be used to enforce backwards compatibility with loss group """
    def dump_config(self, config: dict):
        pass

    @staticmethod
    def load_config(path):
        pass

    from shutil import rmtree
    import pickle

    def __init__(self, **kwargs):
        super(DefaultLossGroup, self).__init__(**kwargs)

    @property
    def res_dir_absolute_path(self):
        return "C:\\Users\Danielle\Google Drive\Shmekels_drive\shmekel_loss_results"

    def get_history_path(self, epoch: int):
        return os.path.join(str(self.experiment), 'history.h5')

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


class DefaultModelGroup(BaseBackupHandler):
    """
    saves and reads file locally on computer.

    definitions:
        -- snapshots are saved and loaded via keras
        -- configs are saved as json files
        -- histories are saved as "h5" files
    """
    import shutil
    import pickle
    import json
    import io

    def __init__(self, **kwargs):
        super(DefaultModelGroup, self).__init__(**kwargs)

    @property
    def res_dir_absolute_path(self):
        return os.path.abspath(os.path.join(os.path.pardir, 'Shmekel_Results'))

    def dump_snapshot(self, model, epoch: int):
        if not os.path.exists(os.path.join(self.exp_absolute_path, self.snapshots_dir_relative_path)):
            os.makedirs(os.path.join(self.exp_absolute_path, self.snapshots_dir_relative_path))
        if epoch == 1:
            model.save(self.get_snapshot_path(epoch))
        else:
            model.save_weights(self.get_snapshot_path(epoch))

    def load_snapshot(self, model, epoch: int):
        if not os.path.exists(self.get_snapshot_path(epoch)):
            return None
        if epoch == 1:
            model = load_model(self.get_snapshot_path(epoch))
        else:
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
        path = str(self.exp_absolute_path)
        self.shutil.rmtree(path)

    def dump_config(self, config: dict):
        path = self.get_config_path()
        dir_path = path.rsplit(os.path.sep, 1)[0]

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with self.io.open(path, 'w', encoding='utf8') as outfile:
            outfile.write(self.json.dumps(config, indent=4, sort_keys=True,
                                          separators=(',', ': '), ensure_ascii=False))

    @staticmethod
    def load_config(path):
        import json
        with open(path, 'r') as f:
            data = f.read()
            c = json.loads(data)
        return c
