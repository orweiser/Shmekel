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