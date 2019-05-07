from .stock_dataset import StocksDataset
import numpy as np
import matplotlib.pyplot as plt
from shmekel_core.calc_models import Stock
from feature_space import get_feature


class SmoothStocksDataset(StocksDataset):

    # def __getitem__(self, index) -> dict:
    #     o_index = index
    #     stock = None
    #     for stock in self.stocks_list:
    #         if index < self.stock_effective_len(stock):
    #             break
    #         index = index - self.stock_effective_len(stock)
    #
    #     inputs = copy(stock.feature_matrix[index: index + self.time_sample_length, :self.num_input_features])
    #     outputs = copy(stock.feature_matrix[index, self.num_input_features:])
    #
    #     item = {'inputs': inputs, 'outputs': outputs, 'id': o_index}
    #     for (s, _), f in zip(self._non_numerical_features, stock.not_numerical_feature_list):
    #         item[s] = f[index]
    #
    #     return item

    @property
    def stocks_list(self):
        if self._stocks_list is None:
            smooth_stocks_list = []
            for stock in super().stocks_list:
                smooth_features_list = []
                for feature in stock.numerical_feature_list[:-1]:
                    smooth_features_list.append(
                        np.convolve(feature, np.ones((self.time_sample_length,)) / self.time_sample_length,
                                    mode='valid'))
                smooth_stocks_list.append(
                    Stock(stock.stock_tckt, smooth_features_list,
                          feature_list=[get_feature(f_name, **params) for f_name, params in
                                        self.feature_list_with_params + self._non_numerical_features]))
            plt.figure()
            for stock in super().stocks_list:
                plt.plot(stock.numerical_feature_list[0])
                break
            for stock in smooth_stocks_list:
                plt.plot(stock.numerical_feature_list[0])
                break
            plt.show()
            self._stocks_list = smooth_stocks_list
        return self._stocks_list
