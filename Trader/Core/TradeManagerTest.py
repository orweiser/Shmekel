from Trader.Core.Manager import *
import pandas as pd

simulation_manager = TradeManager(RunMode.simulate)
simulation_manager.start()
print("done...")

def GetPredictions():
    print("get predictions")