from keras.callbacks import Callback


class BackupHandler(Callback):
    def __init__(self, project: str, experiment,
                 snapshot_backup_delta=1, history_backup_delta=1):
        super(BackupHandler, self).__init__()

        self.experiment = experiment
        self.project = project

        self.snapshot_backup_delta = snapshot_backup_delta
        self.history_backup_delta = history_backup_delta

        self.config = dict(project=project, experiment=experiment,
                           snapshot_backup_delta=snapshot_backup_delta,
                           history_backup_delta=history_backup_delta)

    """ Base Paths: """
    @property
    def project_absolute_path(self):  # todo
        return

    @property
    def exp_relative_path(self):  # todo
        return

    def get_config_path(self):
        return  # todo: join(project_abs_path, exp_rel_path, "config.json"

    """ Handle Snapshots: """
    @property
    def snapshots_dir_relative_path(self):  # todo
        return

    def get_snapshot_path(self, epoch: int):
        return

    def dump_snapshot(self, model, epoch: int):
        raise NotImplementedError()

    def load_snapshot(self, model, epoch: int):
        raise NotImplementedError()

    """ Handle history files: """
    @property
    def histories_dir_relative_path(self):  # todo
        return

    def get_history_path(self, epoch: int):
        return

    def dump_history(self, history: dict, epoch: int):
        raise NotImplementedError()

    def load_history(self, epoch: int):
        raise NotImplementedError()

    """ Callback methods: """
    def on_train_begin(self, logs=None):
        # todo: backup exp config
        self.epochs_done = None

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_done = (epoch + 1)

        if not (epoch + 1) % self.history_backup_delta:
            pass  # todo: backup history

        if not (epoch + 1) % self.snapshot_backup_delta:
            pass  # todo: backup weights

    def on_train_end(self, logs=None):
        epoch = self.epochs_done
        pass  # todo: backup history and weights

    """ More: """
    def erase(self):
        raise NotImplementedError()
