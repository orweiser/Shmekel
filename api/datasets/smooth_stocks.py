from .stock_dataset import StocksDataset
import numpy as np
import matplotlib.pyplot as plt
from shmekel_core.calc_models import Stock
from feature_space import get_feature


class SmoothStocksDataset(StocksDataset):
    figpath: str

    def init(self, config_path=None, time_sample_length=5,
             stock_name_list=None, feature_list=None, val_mode=False, output_feature_list=None, figpath=None):
        super().init(config_path, time_sample_length,
                     stock_name_list, feature_list, val_mode, output_feature_list)
        self.figpath = figpath

    @property
    def stocks_list(self):
        if self._stocks_list is None:
            smooth_stocks_list = []
            for stock in super().stocks_list:
                smooth_stock_data = []
                temp_list = []
                for feature in stock.numerical_feature_list[:-1]:
                    temp_list.append(
                        np.convolve(feature, np.ones((self.time_sample_length,)) / self.time_sample_length,
                                    mode='valid'))
                for i, stock_data in enumerate(temp_list[0]):
                    smooth_stock_data.append(
                        (temp_list[0][i], temp_list[1][i], temp_list[2][i], temp_list[3][i], temp_list[4][i]))
                smooth_stock_data = [np.array(smooth_stock_data), list(range(len(smooth_stock_data)))]
                smooth_stocks_list.append(
                    Stock(stock.stock_tckt, smooth_stock_data,
                          feature_list=[get_feature(f_name, **params) for f_name, params in
                                        self.feature_list_with_params]))
            plt.figure()
            for stock in super().stocks_list:
                plt.plot(stock.numerical_feature_list[0])
                break
            for stock in smooth_stocks_list:
                plt.plot(stock.numerical_feature_list[0])
                break
            plt.title("Smooth stock plot - window size {}".format(self.time_sample_length))
            plt.savefig(self.figpath)
            self._stocks_list = smooth_stocks_list
        return self._stocks_list
