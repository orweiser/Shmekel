from keras import Model as KerasModel
from copy import deepcopy as copy


class Model(KerasModel):
    """
    a wrapper around keras Model class that is compatible with our api
    for example, this class holds the model's config by our parameters

    subclasses must implement the following methods:
        init
        get_inputs_outputs
        __str__

    optional methods for advanced training schemes:
        callbacks
    """
    def __init__(self, experiment=None, **params):
        """
        a constructor method.
        replaced by init() in subclasses.

        :param experiment: optional. an Experiment instance
        :param params: holds all the key-word arguments except for "experiment"
                those key-words are passed to the init() method
        """
        self.experiment = experiment

        self.config = copy(self.get_default_config())
        self.config.update(dict(experiment=experiment, **copy(params)))

        self.init(**params)

        inputs, outputs = self.get_input_output_tensors()

        super(Model, self).__init__(inputs, outputs, name=self.__str__())

    def init(self, *args, **kwargs):
        """
        mimics a constructor. as a constructor, does not return anything.

        to implement the init method, replace the __init__ method as follows:

                def init(self, arg1, arg2, ...)
                    self.arg1 = arg1 ...
                    ...

            instead of:
                def __init__(self, arg1, arg2, ...)
                    self.arg1 = arg1 ...
                    ...
        """
        raise NotImplementedError()

    def get_input_output_tensors(self):
        """
        here lies the architecture of the model
        we are using keras Model API

        this method returns inputs and outputs in the same sense that the keras API works
        for clarity, the following code should have no problem:

            from keras import Model as KerasModel

            inputs, outputs = get_input_output_tensors
            model = KerasModel(inputs, outputs)

        see keras documentation
        or the FullyConnected example
        """
        raise NotImplementedError()

    def __str__(self):
        """
        should return a string with the name of the subclass, including some parameters
                                from config to distinguish between instances
        """
        raise NotImplementedError()

    @property
    def callbacks(self):
        """for later use of advanced training schemes"""
        return []

    def get_default_config(self) -> dict:
        """
        :return: a default configuration dictionary.
            typically it should contain all the keywords from init
        """
        raise NotImplementedError()


