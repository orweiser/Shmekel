from .dataset import Dataset
from .loss import Loss
from .model import Model
from .trainer import Trainer
from .experiment import Experiment
from .backup_handler import get_handler
from Utils.logger import logger


def get_exp_from_config(config: dict):
    return Experiment(**config)


def load_config(path, handler=None):
    import os

    if not os.path.exists(path):
        raise RuntimeError('could not find path: ' + path)

    if os.path.isdir(path):
        new_path = os.path.join(path, 'config.json')
        logger.debug('found a directory at path:', path, '\nLooking for a config file in:', new_path)
        path = new_path

    handler = get_handler(handler or 'DefaultLocal', instantiate=False)
    return handler.load_config(path)

