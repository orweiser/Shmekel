from ..core.dataset import Dataset
from utils.data import DataReader
from shmekel_core.calc_models import Stock
from feature_space import get_feature
from copy import deepcopy as copy
import os


DEFAULT_TRAIN_STOCK_LIST = ('fb',)
DEFAULT_VAL_STOCK_LIST = ('cool',)

DEFAULT_INPUT_FEATURES = (('candle', {'with_volume': False}),)
DEFAULT_OUTPUT_FEATURES = (('rise', dict(output_type='categorical', k_next_candle=1)),)


class StocksDataset(Dataset):
    time_sample_length: int
    _val_mode: bool
    _stocks_list: (list, tuple)
    config_path: str
    output_features: (list, tuple)
    input_features: (list, tuple)
    stock_name_list: (list, tuple)
    feature_list_with_params: (list, tuple)
    _non_numerical_features: (list, tuple)
    _num_input_features: int
    _num_output_features: int

    def init(self, config_path=None, time_sample_length=1,
             stock_name_list=None, feature_list=None, val_mode=False, output_feature_list=None):

        if config_path is None:
            config_path = 'Shmekel_config.txt'
            if not os.path.exists(config_path):
                config_path = os.path.join(os.path.pardir, config_path)
        assert os.path.exists(config_path), 'didnt found file "Shmekel_config.txt", ' \
                                            'please specify the Shmekel config path'

        self.time_sample_length = time_sample_length
        self._val_mode = val_mode
        self._stocks_list = None
        self.config_path = config_path

        self.output_features = output_feature_list or DEFAULT_OUTPUT_FEATURES
        self.input_features = feature_list or DEFAULT_INPUT_FEATURES

        self.stock_name_list = stock_name_list
        self.stock_name_list = self.stock_name_list or \
                               (DEFAULT_VAL_STOCK_LIST if val_mode else DEFAULT_TRAIN_STOCK_LIST)

        self.feature_list_with_params = [
            x if isinstance(x, (tuple, list)) else (x, {}) for x in self.input_features + self.output_features
        ]
        self._non_numerical_features = [('DateTuple', {})]

        self._num_input_features = None
        self._num_output_features = None

    def get_default_config(self) -> dict:
        return dict(config_path=None, time_sample_length=1, stock_name_list=None,
                    feature_list=None, val_mode=False, output_feature_list=None)

    @property
    def num_input_features(self):
        if self._num_input_features is None:
            f_with_params = self.feature_list_with_params[:len(self.input_features)]

            self._num_input_features = sum([get_feature(f_name, **params).num_features
                                            for f_name, params in f_with_params])

        return self._num_input_features

    @property
    def num_output_features(self):
        if self._num_output_features is None:
            f_with_params = self.feature_list_with_params[len(self.input_features):]

            self._num_output_features = sum([get_feature(f_name, **params).num_features
                                            for f_name, params in f_with_params])

        return self._num_output_features

    def stock_effective_len(self, s):
        return len(s) - self.time_sample_length + 1

    def __getitem__(self, index) -> dict:
        o_index = index
        stock = None
        for stock in self.stocks_list:
            if index < self.stock_effective_len(stock):
                break
            index = index - self.stock_effective_len(stock)

        inputs = copy(stock.feature_matrix[index: index + self.time_sample_length, :self.num_input_features])
        outputs = copy(stock.feature_matrix[index, self.num_input_features:])

        item = {'inputs': inputs, 'outputs': outputs, 'id': o_index}
        for (s, _), f in zip(self._non_numerical_features, stock.not_numerical_feature_list):
            item[s] = f[index]

        return item

    @property
    def stocks_list(self):
        if self._stocks_list is None:
            reader = DataReader(self.config_path)

            self._stocks_list = [Stock(tckt, reader.load_stock(tckt),
                feature_list=[get_feature(f_name, **params)
                              for f_name, params in self.feature_list_with_params + self._non_numerical_features])
                                 for tckt in self.stock_name_list]
        return self._stocks_list

    @property
    def val_mode(self) -> bool:
        return self._val_mode

    @property
    def input_shape(self) -> tuple:
        return self.time_sample_length, self.num_input_features

    @property
    def output_shape(self) -> tuple:
        return tuple([self.num_output_features])

    def __len__(self) -> int:
        return sum([self.stock_effective_len(stock) for stock in self.stocks_list])

    def __str__(self) -> str:
        return 'StocksDataSet-' + 'val' * self.val_mode
