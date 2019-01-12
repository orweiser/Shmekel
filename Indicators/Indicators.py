import numpy as np


"""
Hi, in this file we define the general classes of Features and Stocks.
for now, please ignore the StockList class.

in implementing the features, your main focus will be on defining subclasses of Feature. 
the real must do is to redefine the "_compute_feature" method.
go to the end of the file for pseudo-example.

Stock.load_data must be implemented. Rotem said it might be easier to directly download data each time.
    check it out and decide on the best way to load data.
    note that there need to be some notion of train and validation sets

we need to decide on normalization methods. if you get to it, do implement it
but if not it's ok because it should not be feature specific
    
last thing, note that in Stock.build i try to define self.temporal_size, once you do load the data in some pattern,
    make sure to finish the definition there
    
"""


def normalize(feature, normalization_type=None):
    """
    different normalization methods.

    :param feature: a numpy array to normalize
    :param normalization_type: some identification of the normalization type
    :return: a normalized numpy array of the same shape

    note: though normalization is important, we did not decide on types exactly. therefore, this
    method is not necessary at first stage.
    """

    normalization_type = normalization_type or normalization_type
    # todo: add normalization methods

    return feature


class Feature:
    """
    this is the general class of features
    !!! it is NOT a specific feature. specific features will be subclasses of this class !!!

    The code might require some corrections and adaptations, but in general it needs not to be changed

    """
    def __init__(self, data=None, normalization_type=None, time_delay=0, is_numerical=None):
        """
        :param data: None or a pointer to the stock data

        :param normalization_type:

        :param time_delay: an integer that specifies the number of past samples needed to compute
                    the an entry .note that if N samples are required, then the output feature
                    vector is N samples shorter than the original data

        :param is_numerical: a boolean value. True if the feature is numerical and False otherwise.
                    for example, "date" is a non-numerical feature because the usual adding and
                    multiplication does not make sense.
                    However, we might want a different variation on "Date" that is numerical, for
                    example we could define a feature to indicate the day of the week via 1-hot
                    representation - more on that in another time
        """
        self.data = data
        self.normalization_type = normalization_type
        self.is_numerical = is_numerical
        self.time_delay = time_delay

    def __call__(self, *args, **kwargs):
        """Don't mind it, for later use"""
        self.__init__(*args, **kwargs)

    def _compute_feature(self, data):
        """
        This is the core function of the feature. use it to define the feature's functionality
        NOTE: do not edit the code here, instead use inheritance to create sub classes with different
            "_compute_feature" definitions.

        ****This method must be overridden by children classes****

        :param data:
        :return:
        """
        pass

    def get_feature(self, temporal_delay=0, normalization_type=None):
        """
        compute feature on self.data

        ****This method should generally be inherited by children classes****

        :param temporal_delay: an integer.
            if temporal_delay and (temporal_delay < self.time_delay):
                raise Exception('while using method "get_feature" temporal_delay can not be smaller than self.time_delay)
            else:
                drop the oldest (temporal_delay - self.time_delay) values in  the feature

        :param normalization_type: normalization type as explained in the method "normalize".
        :return: a numpy array of shape (Stock_num_samples - time delay, feature size)
        """
        if temporal_delay and (temporal_delay < self.time_delay):
            raise Exception('while using method "get_feature" temporal_delay can not be smaller than self.time_delay')

        if self.data is None:
            return None

        feature = self._compute_feature(self.data)

        feature = normalize(feature, normalization_type=normalization_type)

        if temporal_delay > self.time_delay:
            feature = feature[:-(temporal_delay - self.time_delay)]

        return feature


class Stock:
    """
    this class represents a stock with it's features, holds the data and the computations
    you need to de two things here:
        1. implement the method "load_data"
        2. define self.temporal_size in method "build"
    """
    def __init__(self, stock_tckt, data=None, feature_list=None, normalization_types=None, validation=False):
        """
        :param stock_tckt: just the name of the stock or whatever identifier you decide is best
        :param data: optional if "load_data" is implemented.
        :param feature_list: list of features (Feature subclasses) to compute for the stock
        :param normalization_types: the normalization types to use on each feature.
            if normalization_types is a list, it should be the same length as feature_list.
            else, it is used on all features
        :param validation: boolean. if true, method "load_data" will load validation data instead of train data
        """
        self.stock_tckt = stock_tckt
        self.data = data
        self.validation = validation
        self.features = feature_list if type(feature_list) is list else [feature_list]
        self.normalization_types = normalization_types if type(normalization_types) is list \
            else [normalization_types] * len(self.features)
        self.built = False
        self.feature_matrix = None
        self.temporal_delay = None
        self.temporal_size = None

    def load_data(self):
        """
        loads the data if self.data is None, else it returns self.data
        """
        
        if self.data is None:
            stock = self.stock_tckt
            self.data=np.load('../Data/Stocks_np/' + stock.split('.')[0] + '.us.txt.npy')
        return self.data

    def build(self):
        """
        loads the data and create feature instances
        """
        if self.built:
            return
        data = self.load_data()
        self.features = [feature(data=data, normalization_type=n_type)
                         for feature, n_type in zip(self.features, self.normalization_types)]
        self.temporal_delay = max([feature.time_delay for feature in self.features])
        self.temporal_size = np.shape(data)[0]-self.temporal_delay
        self.built = True

    def get_features_as_list(self, only_numerical_features=False, only_not_numerical_features=False, align=False):
        """
        this method outputs a list of features as numpy arrays
        NOTE: unless :param align: is True, those features are not necessarily the same size

        :param only_numerical_features: return only features that are numerical
        :param only_not_numerical_features: return only features that are not numerical
        :param align: if True, returns numpy arrays of the same temporal size
        :return: a list of numpy arrays
        """
        if only_numerical_features and only_not_numerical_features:
            raise Exception('at least one of the parameters only_not_as_matrix and only_as_matrix_features'
                            'must be False.')
        self.build()

        temporal_delay = 0 if not align else self.temporal_delay

        if only_not_numerical_features:
            return [feature.get_feature(temporal_delay=temporal_delay) for feature in self.features if not feature.is_numerical]
        if only_numerical_features:
            return [feature.get_feature(temporal_delay=temporal_delay) for feature in self.features if feature.is_numerical]
        return [feature.get_feature(temporal_delay=temporal_delay) for feature in self.features]

    def get_features_as_matrix(self, override=False):
        """
        returns the full feature matrix of the stock
        :param override: if True, past values are overridden
        :return: a numpy 2-D array
        """
        if override or self.feature_matrix is None:
            self.feature_matrix = np.stack(self.get_features_as_list(only_numerical_features=True, align=True))

        return self.feature_matrix

    def slice(self, t_start, t_end=None, num_time_samples=None):
        """
        computes a slice of the full feature matrix
        :param t_start: start time index
        :param t_end: optional: end time index. if not specified, t_end = t_start + num_time_samples
        :param num_time_samples: optional
        :return: a numpy 2-D array
        """
        if t_end is None and num_time_samples is None:
            raise Exception('while using "slice", either t_end or num_time_samples must be specified')

        if num_time_samples:
            t_end = t_start + num_time_samples

        return self.get_features_as_matrix()[t_start:t_end]


# class StockList:
#     """
#     NOTE: this class is not yet thought of. ignore it for now.
#     """
#     def __init__(self, stock_tckt_list, features_list, normalization_types=None):
#         def L(x):
#             if type(x) is list:
#                 return copy(x)
#             else:
#                 return [copy(x) for _ in range(num_features)]
#
#         num_features = 1 if type(features_list) is not list else len(features_list)
#
#         self.stock_tckt_list = L(stock_tckt_list)
#         self.normalization_types = L(normalization_types)
#         self.feature_list = L(features_list)
#
#         for f in self.feature_list:
#             if not issubclass(f, Feature):
#                 raise Exception('all features in feature_list must be a subclass of Feature')
#
#         self.built = False
#         self.stock_list = None
#
#     def build(self):
#         if self.built:
#             return
#
#         self.stock_list = []
#         for tckt in self.stock_tckt_list:
#             self.stock_list.append(
#                 Stock(stock_tckt=tckt, feature_list=self.feature_list, normalization_types=self.normalization_types)
#             )
#
#         self.built = True
#
#     def generator(self, batch_size=512, time_length=32, randomize=True):
#         pass


"""Down here we define the features"""


class __feature_example__(Feature):
    """
    this is an example on how to create a feature subclass.
    use this green space to describe the feature for those of us who are not familiar
    """
    def __init__(self, data=None, normalization_type=None):  # DO NOT CHANGE THE DECLARATION
        """
        use this method to define the parameters of the feature
        """

        self.time_delay = 0  # change it according to the feature as described in class Feature
        self.is_numerical = None  # change it according to the feature as described in class Feature

        # here you can define more parameters that "_compute_feature" might need to use
        self.param1 = None
        self.param2 = None
        # ...
        self.paramN = None

        # the following line must be included
        super(__feature_example__, self).__init__(data=data, normalization_type=normalization_type,
                                                  time_delay=self.time_delay, is_numerical=self.is_numerical)

    def _compute_feature(self, data):
        """
        That's the core method of the feature. It MUST be re-defined for every feature

        define the function to output the feature as numpy array from the data

        :param data:
        :return: a numpy array
        """

        # if you defined some extra parameters, you can access them as follows:
        param1 = self.param1    # and so on...

        f = data ** 2  # just some calculations to create the feature

        return f




def smooth_moving_avg(data_seq, period):
    data_seq = np.flip(data_seq)
    smma = np.ndarray(np.size(data_seq) - period + 1)
    smma[0] = np.mean(data_seq[:period])

    for idx in range(period, np.size(data_seq)):
        smma[idx - period + 1] = (1 - 1/period)*smma[idx - period] + (1/period)*data_seq[idx]

    smma = np.flip(smma)
    return smma

# consider changing smma to class instead of a function
# class SMMA(Feature):
#     def __init__(self, time_series=None, period=14, data=None, normalization_type=None):  # DO NOT CHANGE THE DECLARATION
#         """
#         use this method to define the parameters of the feature
#         """
#
#         self.time_delay = period - 1  # change it according to the feature as described in class Feature
#         self.is_numerical = 1  # boolean. change it according to the feature as described in class Feature
#
#         # here you can define more parameters that "_compute_feature" might need to use
#         self.time_series = time_series
#         self.period = period
#
#         # the following line must be included
#         super(SMMA, self).__init__(data=data, normalization_type=normalization_type)


class RSI(Feature):
    def __init__(self, period=14, data=None, normalization_type=None):  # DO NOT CHANGE THE DECLARATION
        """
        use this method to define the parameters of the feature
        """

        self.time_delay = period   # change it according to the feature as described in class Feature
        self.is_numerical = 1  # boolean. change it according to the feature as described in class Feature

        # here you can define more parameters that "_compute_feature" might need to use
        self.period = period
        self.epsilon = 1e-8

        # the following line must be included
        super(RSI, self).__init__(data=data, normalization_type=normalization_type)

    def _compute_feature(self, data):
        """
        data columns represent'Open', 'High', 'Low', 'Close', 'Volume'
                                0       1       2       3         4     
        """
        close = data[:, 3]
        dif = -np.diff(close)
        u = np.maximum(dif, 0)
        d = np.maximum(-dif, 0)

        smmau = smooth_moving_avg(u, self.period)
        smmad = smooth_moving_avg(d, self.period)

        rs = smmau / (smmad + self.epsilon)  # adding epsilon for numerical purposes
        rsi = 100 - 100 / (1 + rs)
        return rsi


class ADL(Feature):
    def __init__(self, data=None, normalization_type=None):  # DO NOT CHANGE THE DECLARATION
        """
        use this method to define the parameters of the feature
        """

        self.time_delay = 0  # change it according to the feature as described in class Feature
        self.is_numerical = 1  # boolean. change it according to the feature as described in class Feature

        # here you can define more parameters that "_compute_feature" might need to use

        # the following line must be included
        super(ADL, self).__init__(data=data, normalization_type=normalization_type)

    def _compute_feature(self, data):
        
        """
        data columns represent'Open', 'High', 'Low', 'Close', 'Volume'
                                0       1       2       3         4     
        """
        close, low, high, volume = data[:, 3], data[:, 2], data[:, 1], data[:, 4]

        AD = volume*(2 * close - low - high)/(high - low)
        # AD = volume*(2 * close + (-low) + (-high))/(high + (-low))

        return AD


class MFI(Feature):
    def __init__(self, period=14, data=None, normalization_type=None):  # DO NOT CHANGE THE DECLARATION
        """
        use this method to define the parameters of the feature
        """

        self.time_delay = period  # change it according to the feature as described in class Feature
        self.is_numerical = 1  # boolean. change it according to the feature as described in class Feature

        # here you can define more parameters that "_compute_feature" might need to use
        self.period = period
        # the following line must be included
        super(MFI, self).__init__(data=data, normalization_type=normalization_type)

    def _compute_feature(self, data):
        """
        data columns represent'Open', 'High', 'Low', 'Close', 'Volume'
                                0       1       2       3         4     
        """
        close, low, high, volume = data[:,3], data[:,2], data[:,1], data[:,4]
    

        typical_price = ( high + low + close)/3
        money_flow = volume*typical_price

        positive_money_flow = np.zeros(1)
        negative_money_flow = np.zeros(1)

        for idx in range(self.period):

            if typical_price[idx] > typical_price[idx+1]:
                positive_money_flow += money_flow[idx]

            if typical_price[idx] < typical_price[idx+1]:
                negative_money_flow += money_flow[idx]

        money_ration = positive_money_flow/negative_money_flow
        money_flow_index = 100 - 100/(1 + money_ration)

        return money_flow_index


class Stochastic(Feature):
    def __init__(self, period=14, data=None, normalization_type=None):  # DO NOT CHANGE THE DECLARATION
        """
        use this method to define the parameters of the feature
        """

        self.time_delay = period + 2  # change it according to the feature as described in class Feature
        self.is_numerical = 1  # boolean. change it according to the feature as described in class Feature

        # here you can define more parameters that "_compute_feature" might need to use
        self.period = period
        # the following line must be included
        super(Stochastic, self).__init__(data=data, normalization_type=normalization_type)

    def _compute_feature(self, data):
        """
        data columns represent'Open', 'High', 'Low', 'Close', 'Volume'
                                0       1       2       3         4     
        """
        close, low, high = data[:,3], data[:,2], data[:,1]

        K =  np.zeros(np.size(close) - self.period + 1)
        for idx in range(np.size(close) - self.period):

            K[idx] = 100*(close[idx] - np.max(low[idx:(idx + self.period)]))/(np.max(high[idx:(idx + self.period)]) - np.max(low[idx:(idx + self.period)]))

        return K[0], (K[0] + K[1] + K[2])/3


class BollingerBands(Feature):  # returns 2 degenerate fetures

    def __init__(self, smoothing_period=20, number_of_SD=2, data=None, normalization_type=None):  # DO NOT CHANGE THE DECLARATION
        """
        use this method to define the parameters of the feature
        """

        self.time_delay = smoothing_period  # change it according to the feature as described in class Feature
        self.is_numerical = 1  # boolean. change it according to the feature as described in class Feature

        # here you can define more parameters that "_compute_feature" might need to use
        self.smoothing_period = smoothing_period
        self.number_of_SD = number_of_SD
        # the following line must be included
        super(BollingerBands, self).__init__(data=data, normalization_type=normalization_type)

    def _compute_feature(self, data):
        """
        data columns represent'Open', 'High', 'Low', 'Close', 'Volume'
                                0       1       2       3         4     
        """
        close, low, high = data[:,3], data[:,2], data[:,1]

        typical_price = (high + low + close)/3
        moving_avg = np.ndarray(np.size(typical_price) - self.smoothing_period)
        std = np.ndarray(np.size(typical_price) - self.smoothing_period)


        for idx in range(0, np.size(typical_price) - self.smoothing_period):
            moving_avg[idx] = np.mean(typical_price[idx:(idx+self.smoothing_period)])
            std[idx] = np.std(typical_price[idx:(idx + self.smoothing_period)])

        return moving_avg + self.number_of_SD*std, moving_avg - self.number_of_SD*std


class ADX(Feature):

    def __init__(self, period=14, data=None, normalization_type=None):  # DO NOT CHANGE THE DECLARATION
        """
        use this method to define the parameters of the feature
        """

        self.time_delay = 2*period - 1  # change it according to the feature as described in class Feature
        self.is_numerical = 1  # boolean. change it according to the feature as described in class Feature

        # here you can define more parameters that "_compute_feature" might need to use
        self.period = period
        # the following line must be included
        super(ADX, self).__init__(data=data, normalization_type=normalization_type)

    def _compute_feature(self, data):
        """
        data columns represent'Open', 'High', 'Low', 'Close', 'Volume'
                                0       1       2       3         4     
        """
        close, low, high = data[:,3], data[:,2], data[:,1]

        up_move = -np.diff(high)
        down_move = np.diff(low)

        true_range = np.amax(np.array([high[:-1] - low[:-1], np.abs(high[:-1] - close[1:]), np.abs(low[:-1] - close[1:])]), axis=0)
        avg_true_range = smooth_moving_avg(true_range, self.period)

        dm_plus = up_move * (up_move > 0) * (up_move > down_move) / avg_true_range
        dm_minus = down_move * (down_move > 0) * (down_move > up_move) / avg_true_range

        di_plus = smooth_moving_avg(dm_plus, self.period)
        di_minus = smooth_moving_avg(dm_minus, self.period)
        adx = 100 * smooth_moving_avg(np.abs((di_plus - di_minus) / (di_plus + di_minus)), self.period)

        return adx


class CCI(Feature):

    def __init__(self, period=14, data=None, normalization_type=None):  # DO NOT CHANGE THE DECLARATION
        """
        use this method to define the parameters of the feature
        """

        self.time_delay = period - 1  # change it according to the feature as described in class Feature
        self.is_numerical = 1  # boolean. change it according to the feature as described in class Feature

        # here you can define more parameters that "_compute_feature" might need to use
        self.period = period
        # the following line must be included
        super(CCI, self).__init__(data=data, normalization_type=normalization_type)

    def _compute_feature(self, data):

        """
        data columns represent'Open', 'High', 'Low', 'Close', 'Volume'
                                0       1       2       3         4     
        """
        
        close, low, high = data[:,3], data[:,2], data[:,1]

        typical_price = (high + low + close) / 3
        sma = smooth_moving_avg(typical_price, self.period)
        mean_abs_val = np.mean(np.abs(typical_price - np.mean(typical_price)))

        cci = (typical_price[:np.size(sma)] - sma)/(0.015*mean_abs_val)

        return cci


class momentum_indicator(Feature):
    def __init__(self, period=14, data=None, normalization_type=None):  # DO NOT CHANGE THE DECLARATION
        """
        use this method to define the parameters of the feature
        """

        self.time_delay = period  
        self.is_numerical = 1  
        

        super(momentum_indicator, self).__init__(data=data, normalization_type=normalization_type, time_delay=period)


    def _compute_feature(self, data):
        """
         momentum indicator compares the current price to selected number of previous prices.
         The function recieve Closer values and return each day (current close value) - sum(previous close values)
         data columns represent'Open', 'High', 'Low', 'Close', 'Volume'
                                0       1       2       3         4   
        """ 
        close=data[:,0]
        N=self.time_delay
        M=np.shape(data)[0]
        mom=np.zeros(M-N-1)
        for i in range(M-N-1):
            mom[i]=close[i+N]-sum(close[i:i+N])   
        return mom


class AccumDest(Feature):
    def __init__(self, period=None, data=None, normalization_type=None):  
        """
        use this method to define the parameters of the feature
        """
        self.is_numerical = 1  

        super(AccumDest, self).__init__(data=data, normalization_type=normalization_type)


    def _compute_feature(self, data):
        """
        A/D (Accumulation/Distribution) indicator is a momentum indicator that attempts to gauge supply and demand
        data columns represent'Open', 'High', 'Low', 'Close', 'Volume'
                                0       1       2       3         4   
        """
        close, low, high, volume = data[:,3], data[:,2], data[:,1], data[:,4]
        M=np.shape(data)[0]
        DA=np.zeros(M)
        DA=np.multiply(np.divide(2*close-low-high,high-low),volume)
        DA=np.cumsum(DA)
        return DA


class VWAP(Feature):
    def __init__(self, period=14, data=None, normalization_type=None):  
        """
        use this method to define the parameters of the feature
        """
        self.time_delay = period
        self.is_numerical = 1  
        

        super(VWAP, self).__init__(data=data, normalization_type=normalization_type, time_delay=period)


    def _compute_feature(self, data):
        """
        VWAP (Volume Weighted Average Price)
        data columns represent'Open', 'High', 'Low', 'Close', 'Volume'
                                0       1       2       3         4     
        """
        close, volume = data[:,3], data[:,4]
        
        N=self.time_delay
        M=np.shape(data)[0]
        vwap=np.zeros(M-N-1)
        vc=np.multiply(close,volume)
        for i in range(M-N-1):
            vwap[i]=np.divide(sum(vc[i:i+N]),sum(volume[i:i+N]))   
        return vwap



"""
run this example to get features (Indicators) for Google stock

Google=Stock('googl', None, [BollingerBands, VWAP, CCI], [None, None, None], False)
IND=Google.get_features_as_list(False, False, False)
print(Google.data) #print stock data (candles) for comparison
print(IND)
"""