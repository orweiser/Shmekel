from ..core.dataset import Dataset
from utils.data import DataReader
from shmekel_core.calc_models import Stock
from feature_space import get_feature


DEFAULT_TRAIN_STOCK_LIST = ('fb',)
DEFAULT_VAL_STOCK_LIST = ('cool',)

DEFAULT_FEATURE_LIST = ('candle',)


def rise(sample, target):
    return 1 if target[0] > sample[0, 0] else 0


class StocksDataSet(Dataset):
    def init(self, config_path=r'C:\Or\Projects\Shmekel_config.txt', stock_name_list=None, time_sample_length=1,
             feature_list=None,
             val_mode=False, output_type='Rise'):
        self.time_sample_length = time_sample_length
        self._val_mode = val_mode
        self._stocks_list = None
        self._config_path = config_path
        self._data_reader = DataReader(config_path)
        self._ouput_type = output_type

        if stock_name_list is None:
            self.stock_name_list = DEFAULT_VAL_STOCK_LIST if val_mode else DEFAULT_TRAIN_STOCK_LIST
        else:
            self.stock_name_list = stock_name_list

        self._feature_list = feature_list or DEFAULT_FEATURE_LIST
        self.feature_list_with_params = [x if isinstance(x, (tuple, list)) else (x, {}) for x in self._feature_list]
        self.num_features = None

    def __getitem__(self, index) -> dict:
        assert index < len(self) and index > 0

        stock = None
        for stock in self.stocks_list:
            if index < len(stock):
                break
            index = index - len(stock)
        # TODO: fix output logic in this class (inputs to outpua function should be the N last features in sample)
        sample = stock.feature_matrix[index - self.time_sample_length: index]
        label = rise(sample, stock.feature_matrix[index + 1])
        return {'inputs': sample, 'outputs': label}

    def get_default_config(self) -> dict:
        return dict(config_path='path_to_config', stock_name_list=None, time_sample_length=1,
             feature_list=None,
             val_mode=False, output_type='Rise')

    @property
    def stocks_list(self):
        if self._stocks_list is None:
            self._stocks_list = [Stock(tckt, self._data_reader.load_stock(tckt), feature_list=[
                get_feature(f_name, **params) for f_name, params in self.feature_list_with_params])
                                 for tckt in self.stock_name_list]
        return self._stocks_list

    @property
    def val_mode(self) -> bool:
        return self._val_mode

    @property
    def input_shape(self) -> tuple:
        if self.num_features is None:
            self.num_features = [get_feature(f_name, **params) for f_name, params in
                                 self.feature_list_with_params]
            self.num_features = sum([f.num_features for f in self.num_features])
        return (self.time_sample_length, self.num_features)

    @property
    def output_shape(self) -> tuple:
        assert self._ouput_type.lower() == 'rise'
        return (1,)

    def __len__(self) -> int:
        return sum([len(stock) for stock in self.stocks_list])

    def __str__(self) -> str:
        return 'StocksDataSet-' + 'val' * self.val_mode
