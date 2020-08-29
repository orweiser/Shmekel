import pandas as pd
import numpy as np
from feature_space_2020.RSI import RSI
from feature_space_2020.SMA import SMA
from feature_space_2020.momentum import MOMENTUM
from feature_space_2020.CCI import CCI
from feature_space_2020.MACD import MACD
from shmekel_core import Stock
from numpy import genfromtxt

# source_root = "D:\shmekels\downloads\price-volume-data-for-all-us-stocks-etfs\Data\Stocks\\"
# destination_root = "D:\\shmekels\\uploads\\feature_test\\"
destination_root = "C:\\Users\\Rotem\\Shmekel\\feature_test\\"
source_root = "c:\\Users\\Rotem\\Shmekel\\stocks_data\\Stocks\\"


def TestStock(stock_name):
    source_file = source_root + stock_name + ".us.txt"
    destination_file = destination_root + stock_name + ".csv"

    print("loading stock:")
    table = pd.read_csv(source_file)

    #add sma
    sma_ranges = [10,20,50]
    for sma_range in sma_ranges:
        label = 'SMA_' + str(sma_range)
        sma = GetSma(source_file,sma_range)
        table[label] = sma
        print("adding "+label)

    #add rsi
    rsi_ranges = [7,14,21]
    for rsi_range in rsi_ranges:
        label = 'RSI_'+str(rsi_range)
        rsi = GetRsi(source_file,rsi_range)
        table[label] = rsi
        print("adding " + label)

    # add momentum
    mom_ranges = [7, 14, 21]
    for mom_range in mom_ranges:
        label = 'momentum_' + str(mom_range)
        mom = GetMom(source_file, mom_range)
        table[label] = mom
        print("adding " + label)

    # add cci
    cci_ranges = [20,]
    for cci_range in cci_ranges:
        label = 'cci_' + str(cci_range)
        cci = GetCci(source_file, cci_range)
        table[label] = cci
        print("adding " + label)

    # add mcad
    label = 'macd_with_signal'
    macd_with_signal = GetMacd(source_file, calc_signal_line=True)
    table[label + '_macd_feature'] = macd_with_signal[:, 0]
    table[label + '_signal_feature'] = macd_with_signal[:, 1]

    label = 'macd_without_signal'
    macd = GetMacd(source_file, calc_signal_line=False)
    table[label + '_macd_feature'] = macd


    table.to_csv(destination_file)


def GetSma(srcFile,range):
    close = genfromtxt(srcFile, delimiter=',',usecols=(4))
    sma = SMA(period=range)
    return sma.process(close[1:])


def GetRsi(srcFile,range):
    close = genfromtxt(srcFile, delimiter=',', usecols=(4))
    rsi = RSI(period=range)
    return rsi.process(close[1:])


def GetMom(srcFile,range):
    close = genfromtxt(srcFile, delimiter=',', usecols=(4))
    mom = MOMENTUM(period=range)
    return mom.process(close[1:])


def GetCci(srcFile, range):
    high = genfromtxt(srcFile, delimiter=',', usecols=(2))
    low = genfromtxt(srcFile, delimiter=',', usecols=(3))
    close = genfromtxt(srcFile, delimiter=',', usecols=(4))
    cci = CCI(range=range)
    return cci.process(high[1:], low[1:], close[1:])


def GetMacd(srcFile, calc_signal_line):
    close = genfromtxt(srcFile, delimiter=',', usecols=(4))
    macd = MACD(calc_signal_line=calc_signal_line)
    return macd.process(close[1:])

TestStock("fbc")
