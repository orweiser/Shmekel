from copy import deepcopy as copy
from utils.logger import logger


class Loss:
    """
    Abstract class for losses

    subclasses should implement the following methods:
        init
        get_default_config
        __str__
        loss

    optional methods to implement for advanced training techniques:
        loss_weights
        callbacks
    """
    def __init__(self, loss='', experiment=None, **params):
        """
        a constructor method.
        replaced by init() in subclasses.

        :type loss: str
        :param loss: a string identifier of Loss subclass
        :param experiment: optional. an Experiment instance
        :param params: holds all the key-word arguments except for "experiment"
                those key-words are passed to the init() method
        """

        if not loss: logger.warning('Loss parameter "loss" is missing')

        self.experiment = experiment

        self.config = copy(self.get_default_config())
        self.config.update(dict(loss=loss, **copy(params)))

        self.init(**params)

        self.y_true_tensor = None
        self.y_pred_tensor = None

        self.loss_tensor = None

    def init(self, *args, **kwargs):
        """
        mimics a constructor. as a constructor, does not return anything.

        to implement the init method, it should mimic the constructor __init__
            it should look like this:
                def init(self, arg1, arg2, ...)
                    self.arg1 = arg1 ...
                    ...

            instead of:
                def __init__(self, arg1, arg2, ...)
                    self.arg1 = arg1 ...
                    ...
        """
        raise NotImplementedError()

    def get_default_config(self) -> dict:
        """
        :return: a default configuration dictionary.
            typically it should contain all the keywords from init
        """
        raise NotImplementedError()

    def __str__(self) -> str:
        """
        should return a string with the name of the subclass, including some parameters
                                from config to distinguish between instances
        """
        raise NotImplementedError()

    def loss(self, y_true, y_pred):
        """
        the loss function to be used by keras
        """
        raise NotImplementedError()

    def __call__(self, y_true, y_pred):
        """
        a wrapper around loss, do not inherit this method
        """
        self.y_true_tensor = y_true
        self.y_pred_tensor = y_pred

        self.loss_tensor = self.loss(y_true, y_pred)

        return self.loss_tensor

    @property
    def loss_weights(self) -> (dict, None):
        """for later use of multiple losses"""
        return None

    @property
    def callbacks(self):
        """for later use of advanced training schemes"""
        return []
