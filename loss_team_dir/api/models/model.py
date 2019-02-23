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
    
        super(Model, self).__init__(*get_layers(), name=self.generate_name())

    @property
    def config(self):
        return copy(self._config)

    @config.setter
    def config(self, value):
        raise RuntimeError('resetting a Models config is not allowed')

    def generate_name(self):
        """:rtype: str"""
        # todo: implement
        return 'model_try'

    @property
    def default_config(self):
        """:rtype: dict"""
        # todo: implement
        return {}

    @property
    def _building_function(self):
        if self.config['model'] == 'fully_connected':
            from .models import fully_connected
            return fully_connected

    @property
    def callbacks(self):
        # todo: implement
        return []

