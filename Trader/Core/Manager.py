import os

from Trader.Core.enums import *
from Trader.Core.Asset import *
from Trader.Core.Candle import *
from Trader.Core.Trade import *
import pandas as pd


class TradeManager:
    def __init__(self, run_mode=RunMode.simulate):
        self.mode = run_mode
        self.assets = []
        self.funds = 0

    def start(self):
        self.get_funds()
        self.load_assets()
        self.run_assets()

    def get_funds(self):
        if self.mode == RunMode.simulate:
            self.funds = Simulation.initial_funds

    def load_assets(self):
        if self.mode == RunMode.simulate:
            self.assets = load_assets_from_folder(Location.stocks_root, 10)

    def run_assets(self):
        if self.mode == RunMode.simulate:
            for asset in self.assets:
                config = TradeConfig
                asset.simulate(trade_config=config)



def load_assets_from_folder(stocks_folder,limit=0,runmode = RunMode.simulate):
    assets = []
    counter = 1
    for dir in os.listdir(stocks_folder):
        sector = dir
        sector_path = stocks_folder+"\\"+sector
        for filename in os.listdir(sector_path):
            filepath = sector_path+"\\"+filename
            name = filename.split('.')[0]
            raw_candles = load_candles(filepath)
            stock = Asset(name, AssetType.stock, sector, raw_candles)
            #stock.calculate_features()
            stock.load_predictors(runmode=runmode)
            assets.append(stock)
            print("added new asset: " + name)
            counter += 1
            if limit > 0:
                if counter > limit:
                    break
        if limit > 0:
            if counter > limit:
                break

        if limit > 0:
            if counter > limit:
                break
    return assets

def load_candles(file_path):
    #print("loading stock from file: " + file_path)

    table = pd.read_csv(file_path)

    #print("loading candles")
    candles = []
    for index, row in table.iterrows():
        candle = Candle(row['Date'],
                        row['Open'],
                        row['High'],
                        row['Low'],
                        row['Close'],
                        row['Volume'])
        candles.append(candle)
    return candles


