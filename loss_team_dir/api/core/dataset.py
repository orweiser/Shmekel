from copy import deepcopy as copy


class Dataset:
    def __init__(self, experiment=None, **params):
        self.experiment = experiment

        self._config = copy(self.get_default_config())
        self._config.update(dict(experiment=experiment, **copy(params)))

        self.init(**params)

    def init(self, *args, **kwargs):
        raise NotImplementedError()

    def get_default_config(self) -> dict:
        raise NotImplementedError()

    def __getitem__(self, index) -> dict:
        """
        return {
                'inputs': inputs,
                'outputs': outputs,

                optional field:
                'stock_name': ...
            }
        """
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()

    # todo: optional methods and properties
    #  train / val split
    #  raw data / processed data split
    #  similar to above: process / deprocess methods
    #  method to get specific items by date or stock name etc.




