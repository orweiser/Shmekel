from copy import deepcopy as copy
from Utils.logger import logger


class BaseDataset:
    """
    Abstract class for datasets.

    subclasses should implement the following abstract methods:
        __get_item__
        __len__
        __str__
    """
    def __getitem__(self, index):
        """
        a method to get dataset items by index
        it should support every index in range(len(self))
        """
        raise NotImplementedError()

    def __len__(self) -> int:
        """
        the number of samples in the dataset.
        method __getitem__() should support any index between 0 and the output of __len__()_
        """
        raise NotImplementedError()

    def __str__(self) -> str:
        """
        should return a string with the name of the subclass,
        including some parameters from config to distinguish between instances
        """
        raise NotImplementedError()


class Dataset(BaseDataset):
    """
    Abstract class for datasets to use during train.

    subclasses should implement the following methods:
        init
        __get_item__ -> returns an input-output pair
        __len__
        __str__
        get_default_config

        input_shape
        output_shape

        val_mode -> returns a boolean. can not be changed after an instance was created
    """
    def __init__(self, dataset, experiment=None, **params):
        """
        a constructor method.
        replaced by init() in subclasses.

        :type dataset: str
        :param dataset: a string identifier of Dataset subclass

        :param experiment: optional. an Experiment instance
        :param params: holds all the key-word arguments except for "experiment"
                those key-words are passed to the init() method
        """
        if not dataset: logger.warning('Dataset parameter "dataset" is missing')
        self.experiment = experiment

        self.config = copy(self.get_default_config())
        self.config.update(dict(dataset=dataset, **copy(params)))

        self.init(**params)

    def init(self, **kwargs):
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

    def __getitem__(self, index) -> dict:
        """
        get the index'th sample of the dataset.
        a sample should be a dictionary with the fields 'inputs', 'outputs'
            where 'inputs' should contain a numpy array of the input data (a single sample)
            and 'outputs' should contain a numpy array of the labels (a single sample)
                * do not account for batches here

        optionally, the dictionary can contain additional fields, e.g. stock name and candle's date

        return {
                'inputs': inputs,
                'outputs': outputs,

                optional fields:
                'stock_name': ...
                ...
            }
        """
        raise NotImplementedError()

    def get_default_config(self) -> dict:
        """
        :return: a default configuration dictionary.
            typically it should contain all the keywords from init
        """
        raise NotImplementedError()

    @property
    def val_mode(self) -> bool:
        """
        when True, __getitem__ returns sample from validation set
                __len__() return the size of the validation set

        this property should be taken care of inside the init method,
            possibly as follows:

                def init(self, ..., val_mode, ...)
                    ...
                    self._val_mode = val_mode

                @property
                def val_mode(self) -> bool:
                    return self._val_mode

        """
        raise NotImplementedError()

    @val_mode.setter
    def val_mode(self, value):
        """
        by design, setting val_mode on and off is forbidden
        the correct methodology is to have to instances of the same dataset,
        one with val_mode = True and the other with False

        possibly as follows:
            train_dataset = DatasetSubclass(..., val_mode=False, ...)
            val_dataset = DatasetSubclass(..., val_mode=True, ...)
        """
        raise AssertionError('Asserting validation mode after initialization is forbidden ')

    @property
    def input_shape(self) -> tuple:
        """returns the shape of sample inputs"""
        raise NotImplementedError()

    @property
    def output_shape(self) -> tuple:
        """returns the shape of sample outputs"""
        raise NotImplementedError()

    # todo: optional methods and properties
    #  train / val split
    #  similar to above: process / deprocess methods
    #  methods: get_item_by_date / get_item_by_stock, etc.
