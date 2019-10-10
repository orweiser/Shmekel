from shmekel_core import Feature
import numpy


class CandleDerivatives(Feature):
    def __init__(self, deriv_order=1, with_volume=True, **kwargs):
        """
        This feature computes the time derivatives (by time scale of data) of a stocks candle
        :param deriv_order: Max order of derivative to compute
        :type deriv_order: int
        :param with_volume: To use volume data (True) of not (False)
        :type with_volume: bool
        :param kwargs:
        """
        super(CandleDerivatives, self).__init__(**kwargs)

        self._deriv_order = deriv_order
        self._num_basic = int(len(self._pattern) - 1*(not with_volume))

        self.is_numerical = True
        self.time_delay = self._deriv_order
        self.with_volume = with_volume
        self.num_features = self._num_basic * self._deriv_order

    def _compute_feature(self, data, feature_list=None):
        data_mat = data[0]
        out_mat = numpy.zeros([data_mat.shape[0], self.num_features])
        for i in range(1, self._deriv_order + 1):
            # TODO: think about using numpy.diff or numpy.gradient
            deriv_mat = data_mat[:-1] - data_mat[1:]
            out_mat[:-i, (i-1)*self._num_basic: i * self._num_basic] = deriv_mat
            data_mat = deriv_mat

        return out_mat[:-self._deriv_order]


