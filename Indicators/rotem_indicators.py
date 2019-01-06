from Dstruct import *
import numpy as np




class momentum_indicator(Feature):
    def __init__(self, period=14, data=None, normalization_type=None):  # DO NOT CHANGE THE DECLARATION
        """
        use this method to define the parameters of the feature
        """

        self.time_delay = period  
        self.is_numerical = 1  
        

        super(momentum_indicator, self).__init__(data=data, normalization_type=normalization_type)


    def _compute_feature(self, data):
        """
         momentum indicator compares the current price to selected number of previous prices.
         The function recieve Closer values and return each day (current close value) - sum(previous close values)
        """ 
        close=np.array(data['Close'])
        N=self.time_delay
        M=np.shape(data)[0]
        mom=np.zeros(M-N-1)
        for i in range(M-N-1):
            mom[i]=closea[i+N]-sum(close[i:i+N])   
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
        """
        close=np.array(data['Close']), low=np.array(data['Low']), high=np.array(data['High']), volume=np.array(data['Volume'])
        M=np.shape(data)[0]
        DA=np.zeros(M)
        DA=np.multiply(np.divide(2*close-low-high,high-low),volume)
        DA=np.cumsum(DA)
        return DA


class VWAP(Feature):
    def __init__(self, period=None, data=None, normalization_type=None):  
        """
        use this method to define the parameters of the feature
        """
        self.time_delay = period
        self.is_numerical = 1  
        

        super(VWAP, self).__init__(data=data, normalization_type=normalization_type)


    def _compute_feature(self, data):
        """
        VWAP (Volume Weighted Average Price)   
        """
        close=np.array(data['Close']), volume=np.array(data['Volume'])
        N=self.time_delay
        M=np.shape(data)[0]
        vwap=np.zeros(M-N-1)
        vc=np.multiply(close,volume)
        for i in range(M-N-1):
            vwap[i]=np.divide(sum(vc[i:i+N]),sum(volume[i:i+N]))   
        return vwap