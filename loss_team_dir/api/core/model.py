from keras import Model as KerasModel
from copy import deepcopy as copy


class Model(KerasModel):
    def __init__(self, experiment=None, **params):
        self.experiment = experiment

        self._config = copy(self.get_default_config())
        self._config.update(dict(experiment=experiment, **copy(params)))

        self.init(**params)

        inputs, outputs = self.get_inputs_outputs()

        super(Model, self).__init__(inputs, outputs, name=self.__str__())

    def init(self, *args, **kwargs):
        raise NotImplementedError()

    def get_inputs_outputs(self):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()

    @property
    def callbacks(self):
        return []

    def get_default_config(self):
        """:rtype: dict"""
        # todo: implement
        return {}


