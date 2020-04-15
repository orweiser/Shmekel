import pandas as pd
import numpy as np
from feature_space_2020.RSI import RSI
from feature_space_2020.SMA import SMA
from shmekel_core import Stock
from numpy import genfromtxt

source_root = "D:\shmekels\downloads\price-volume-data-for-all-us-stocks-etfs\Data\Stocks\\"
destination_root = "D:\\shmekels\\uploads\\feature_test\\"

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

    table.to_csv(destination_file)

def GetSma(srcFile,range):
    close = genfromtxt(srcFile, delimiter=',',usecols=(4))
    sma = SMA(period=range)
    return sma.process(close[1:])

def GetRsi(srcFile,range):
    close = genfromtxt(srcFile, delimiter=',', usecols=(4))
    rsi = RSI(period=range)
    return rsi.process(close[1:])




#RSI DEBUGGIN
def GetRsiBench(srcFile,range):
    close = genfromtxt(srcFile, delimiter=',', usecols=(4))
    rsi = RSI(period=range)
    return rsi.process_bench(close[1:])

def TestRsi(stock_name,range):
    source_file = source_root + stock_name + ".us.txt"
    destination_file = destination_root + stock_name + ".csv"

    print("loading stock:")
    table = pd.read_csv(source_file)
    rsi = GetRsiBench(source_file, range)
    #print(rsi)

def rsi_sma(x,range):
    smma_out = []
    init_avg = np.average(x[0:range])
    smma_out.append(init_avg)
    multiplyer = range-1
    for xval in x[range:]:
        last_val = smma_out[-1]
        new_val = (multiplyer*last_val+xval) / range
        smma_out.append(new_val)
        #print(new_val)
    return np.asarray(smma_out)


#TestRsi("fb",7)
#TestStock("fb")
a = np.random.randint(5, size=(1, 25))
print(a)
print(rsi_sma(a,7))
