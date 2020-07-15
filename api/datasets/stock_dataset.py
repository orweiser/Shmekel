from ..core.dataset import Dataset
from Utils.data import load_stock, get_stock_path
import constants
from shmekel_core.calc_models import Stock
from feature_space import get_feature
from copy import deepcopy as copy
from tqdm import tqdm
from Utils.logger import logger
import os

DEFAULT_TRAIN_STOCK_LIST = [
    'aapl', 'adbe', 'adsk',
    'akam', 'algt', 'amwd',
    'anss', 'asml', 'atni',
    'avgo', 'bcpc', 'bldp',
    'casy', 'chrw', 'cmpr',
    'cohr', 'cprt', 'crmt',
    'csgp', 'cswi', 'csx',
    'cvco', 'cwst', 'dmlp',
    'eslt', 'exls', 'expd',
    'fang', 'fb', 'fele',
    'fwrd', 'goog', 'googl',
    'hele', 'holi', 'hwkn',
    'ilmn', 'intu', 'iosp',
    'irbt', 'jbht', 'klac',
    'lfus', 'lnt', 'lrcx',
    'lstr', 'meli', 'mgee',
    'mksi', 'mlab', 'mpwr',
    'msex', 'msft', 'nblx',
    'ndsn', 'nice', 'novt',
    'ntes', 'nvda', 'odfl',
    'oled', 'ottr', 'patk',
    'pdce', 'pnrg', 'pool',
    'pvac', 'pypl', 'rgld',
    'roll', 'rusha', 'ryaay',
    'saia', 'shen', 'srcl',
    'stmp', 'tmus', 'tsla',
    'uslm', 'vrsn', 'wdfc',
    'wwd', 'xel', 'yorw',
    'zbra', 'zg'
]
DEFAULT_VAL_STOCK_LIST = [
    'abmd', 'algn', 'amzn', 'atri', 'biib',
    'cacc', 'chtr', 'cme', 'coke', 'colm',
    'cost', 'ctas', 'dxcm', 'eqix', 'erie',
    'fcnca', 'iac', 'idxx', 'isrg', 'jbss',
    'jjsf', 'jout', 'lanc', 'lulu', 'mktx',
    'nflx', 'nwli', 'orly', 'pep', 'regn',
    'safm', 'sbac', 'sivb', 'tech', 'tree',
    'uhal', 'vrtx', 'wltw'
]


DEFAULT_TRAIN_YEARS = [x for x in range(1900, 2020) if x % 2]
DEFAULT_VAL_YEARS = [x for x in range(1900, 2020) if not x % 2]


class StocksDataset(Dataset):
    time_sample_length: int
    # normalization_window: int
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
    years: list
    _relevant_indices: dict
    basedir: str
    stocks_ext: str

    def init(self, basedir=None, time_sample_length=5, stocks_ext='us.txt',
             input_features=None, output_features=None,
             stock_name_list=None, val_mode=False,
             years=None, normalization_window=None,
             feature_list=None, output_feature_list=None, config_path=None):

        if feature_list:
            logger.warning('parameter "%s" is deprecated. use "%s" instead' % ('feature_list', 'input_features'))
            input_features = feature_list

        if output_feature_list:
            logger.warning('parameter "%s" is deprecated. use "%s" instead' % ('output_feature_list', 'output_features'))
            output_features = output_feature_list

        if config_path:
            logger.warning('parameter "%s" is deprecated. use "%s" instead' % ('config_path', 'basedir'))
            basedir = config_path

        self.basedir = basedir or constants.DATA_PATH
        self.stocks_ext = stocks_ext

        if normalization_window is not None:
            # todo: fix bug
            #  needs to trim outputs to be the same length as inputs.
            #  do not change k_next_candle to zero just to get over the bug
            #  also "normalization_window" is kind of an ambiguous name for a parameter
            #  also also, consider removing this from the init function of the dataset alltogether.
            #  just send the list below in your config instead
            # default_input_features = (
            # ('High', dict(normalization_type='convolve', normalization_window=normalization_window)),
            # ('Open', dict(normalization_type='convolve', normalization_window=normalization_window)),
            # ('Low', dict(normalization_type='convolve', normalization_window=normalization_window)),
            # ('Close', dict(normalization_type='convolve', normalization_window=normalization_window)),
            # ('Volume', dict(normalization_type='convolve', normalization_window=normalization_window)))
            # self.normalization_window = normalization_window
            raise NotImplementedError('convolve normalization is not supported. please fix the bug in "convolve"')
        default_input_features = ('High', 'Open', 'Low', 'Close', 'Volume')
        default_output_features = (('rise', dict(output_type='categorical', k_next_candle=1)),)

        self.time_sample_length = time_sample_length
        self._val_mode = val_mode
        self._stocks_list = None
        self.config_path = basedir
        self._relevant_indices = {}

        self.output_features = output_features or default_output_features
        self.input_features = input_features or default_input_features

        self.stock_name_list = stock_name_list or (DEFAULT_VAL_STOCK_LIST if val_mode else DEFAULT_TRAIN_STOCK_LIST)

        self.feature_list_with_params = [
            x if isinstance(x, (tuple, list)) else (x, {}) for x in list(self.input_features) + list(self.output_features)
        ]

        self._non_numerical_features = [('DateTuple', {}), ('RawCandle', {})]

        self._num_input_features = None
        self._num_output_features = None

        self.years = None if years is None else sorted(years)

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

    def get_stock_possible_indices(self, s):
        if self.years is None:
            return [i for i in range(len(s) - self.time_sample_length + 1)]

        divided_years_indices_dict = {}
        for i, date_tuple in enumerate(s.not_numerical_feature_list[0]):
            divided_years_indices_dict.setdefault(date_tuple[0], []).append(i)

        last_year = -2
        groups = []
        for year in self.years:
            if year == last_year + 1:
                groups[-1].extend(divided_years_indices_dict.get(year, []))
            else:
                groups.append(divided_years_indices_dict.get(year, []))
            last_year = year

        indices_groups = []
        for group_indices in groups:
            if (len(group_indices) - self.time_sample_length + 1) > 0:
                indices_groups.append(group_indices[:-self.time_sample_length + 1])

        return sum(indices_groups, [])

    def stock_effective_len(self, s):
        if s.stock_tckt not in self._relevant_indices:
            self._relevant_indices[s.stock_tckt] = self.get_stock_possible_indices(s)

        return len(self._relevant_indices[s.stock_tckt])

    def stock_and_local_index_from_global_index(self, index):
        stock = None
        for _stock in self.stocks_list:
            if index < self.stock_effective_len(_stock):
                stock = _stock
                break
            index = index - self.stock_effective_len(_stock)

        if stock is None:
            raise IndexError

        return self._relevant_indices[stock.stock_tckt][index], stock

    def __getitem__(self, o_index) -> dict:
        index, stock = self.stock_and_local_index_from_global_index(o_index)

        inputs = copy(stock.feature_matrix[index: index + self.time_sample_length, :self.num_input_features])
        outputs = None
        try:
            outputs = copy(stock.feature_matrix[index + self.time_sample_length - 1, self.num_input_features:])
        except:
            print("An exception occurred")

        item = {'inputs': inputs, 'outputs': outputs, 'id': o_index, 'stock': stock, '_id': index}
        for (s, _), f in zip(self._non_numerical_features, stock.not_numerical_feature_list):
            item[s] = f[index + self.time_sample_length - 1]

        return item

    def gen_all_paths(self):
        for tckt in tqdm(self.stock_name_list):
            yield tckt, get_stock_path(self.basedir, tckt, self.stocks_ext)

    @property
    def stocks_list(self):
        if self._stocks_list is None:
            logger.info('Loading Stocks:')
            self._stocks_list = []
            for tckt, path in self.gen_all_paths():
                self._stocks_list.append(
                    Stock(tckt, load_stock(path),
                          feature_list=[get_feature(f_name, **params) for f_name, params in self.feature_list_with_params + self._non_numerical_features]))
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

    def __bool__(self) -> bool:
        return bool(self.stock_name_list)


class InferenceStocksDataset(StocksDataset):
    def __init__(self, basedir, **kwargs):
        super(InferenceStocksDataset, self).__init__(**kwargs)
        self.basedir = basedir

    def gen_all_paths(self):
        for tckt_with_ext in os.listdir(self.basedir):
            path = os.path.join(self.basedir, tckt_with_ext)
            tckt = tckt_with_ext.split('.')[0]

            yield tckt, path


