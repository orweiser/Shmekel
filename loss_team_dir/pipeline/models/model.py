from keras import Model as KerasModel
from copy import deepcopy as copy


class Model(KerasModel):
    def __init__(self, experiment=None, model='fully_connected', output_activation='softmax', input_shape=None,
                 **params):
        self._config = self.default_config
        self._config.update({
            **dict(model=model, output_activation=output_activation, input_shape=input_shape, ), **params
        })
        self.experiment = experiment

        def get_layers():
            d = self._building_function()
            return [d[key] for key in ['inputs', 'outputs']]

        super(Model, self).__init__(*get_layers(), name=self.name)

    @property
    def config(self):
        return copy(self._config)

    @config.setter
    def config(self, value):
        raise RuntimeError('resetting a Models config is not allowed')

    @property
    def name(self):
        """:rtype: str"""
        raise NotImplementedError

    @property
    def default_config(self):
        """:rtype: dict"""
        raise NotImplementedError()

    @property
    def _building_function(self):
        return lambda: {}

