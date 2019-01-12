import sys, os
from Indicators import Indicators as ind
import keras

features = [ind.RSI, ind.ADL, ind.MFI, ind.Stochastic, ind.BollingerBands, ind.CCI, ind.momentum_indicator,
            ind.AccumDest, ind.VWAP]    # ind.ADX, need to be added after fix
Google = ind.Stock("googl", None, features)
Google.load_data()
print(Google.data)
print(Google.get_features_as_list(False, False, True))


