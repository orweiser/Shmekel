from sklearn.decomposition import PCA, FastICA


class LinearDimReduction:
    _allowed_types = ['PCA', 'ICA']

    def __init__(self, num_of_components, dim_reduction_type='PCA'):
        self.num_of_components = num_of_components  # todo: add protections on num_of_components value
        self.dim_reduction_type = self.assert_valid_type(dim_reduction_type)

    def assert_valid_type(self, dim_reduction_type):
        if dim_reduction_type not in self._allowed_types:
            raise AssertionError("%s is not a valid dimension reduction type" % dim_reduction_type)
        return dim_reduction_type

    def __call__(self, data):
        if self.dim_reduction_type == 'PCA':
            return self.do_pca(data)
        if self.dim_reduction_type == 'ICA':
            return self.do_ica(data)

    def do_pca(self, data):
        pca = PCA(n_components=self.num_of_components)
        return pca.fit_transform(data)

    def do_ica(self, data):
        fast_ica = FastICA(n_components=self.num_of_components)
        return fast_ica.fit_transform(data)
