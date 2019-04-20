from sklearn.decomposition import PCA, FastICA


class DimReductionConfig:
    _allowed_types = ['PCA', 'ICA']

    def __init__(self, num_of_components, do_dim_reduction=False, dim_reduction_type='PCA'):
        self.do_dim_reduction = do_dim_reduction
        self.num_of_components = num_of_components  # todo: add protections on num_of_components value
        self.dim_reduction_type = self.assert_valid_type(dim_reduction_type)

    def assert_valid_type(self, dim_reduction_type):
        if dim_reduction_type not in self._allowed_types:
            raise AssertionError("%s is not a valid dimension reduction type" % dim_reduction_type)
        return dim_reduction_type


def reduce_dimensionality(data, dim_reduction_config):
    if not dim_reduction_config.do_dim_reduction:
        return data
    if dim_reduction_config.dim_reduction_type == 'PCA':
        return do_pca(data, dim_reduction_config.num_of_components)
    if dim_reduction_config.dim_reduction_type == 'ICA':
        return do_ica(data, dim_reduction_config.num_of_components)


def do_pca(data, num_of_components):
    pca = PCA(n_components=num_of_components)
    return pca.fit_transform(data)


def do_ica(data, num_of_components):
    fast_ica = FastICA(n_components=num_of_components)
    return fast_ica.fit_transform(data)
