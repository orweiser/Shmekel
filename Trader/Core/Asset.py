import os
from random import uniform
from Trader.utilities.sandbox import Prediction
from Trader.Core.enums import *
from feature_space_2020.SMA import *
from feature_space_2020.RSI import *
from Trader.Core.Candle import *
from Trader.Core.Trade import TradeConfig, Trade


def initialize_predictors():
    print("initialize predictors")


class Asset:
    def __init__(self, name, asset_type, sector, candles=[]):
        self.name = name
        self.type = asset_type
        self.sector = sector
        self.candles = candles
        self.predictors = []
        self.open_trades = []
        self.pending_trades = []

    def calculate_features(self):
        rsi_ranges = [14, 21]
        sma_ranges = [10, 25, 50]
        # open = [o.open for o in self.candles]
        # high = [o.high for o in self.candles]
        # low = [o.low for o in self.candles]
        close = [o.close for o in self.candles]

        for rsi_range in rsi_ranges:
            rsi = RSI()
            rsi.range = rsi_range
            rsi.auto_fill = True
            rsi_out = rsi.process(close)

            for i, candle in enumerate(self.candles):
                candle.features["RSI_" + str(rsi_range)] = rsi_out[i]

        for sma_range in sma_ranges:
            sma = SMA()
            sma.range = sma_range
            sma.auto_fill = True
            sma_out = sma.process(close)

            for i, candle in enumerate(self.candles):
                candle.features["SMA_" + str(sma_range)] = sma_out[i]

        print(self.candles[0].features)

    def load_predictors(self, runmode):
        if runmode == RunMode.simulate:
            self.candles = fake_predict_candles(self.candles)

    def simulate(self, trade_config=TradeConfig):
        print("simulation start for Asset: " + self.name)
        simulation_log = []

        simulation_log.append("Asset: " + self.name)
        simulation_log.append(trade_config.LogPrint(trade_config))

        for i, candle in enumerate(self.candles):
            simulation_log.append("\n***********************************************************************\n")
            simulation_log.append("candle num: " + str(i))
            simulation_log.append("datetime: " + str(candle.datetime))
            simulation_log.append("open: " + str(candle.open))
            simulation_log.append("high: " + str(candle.high))
            simulation_log.append("low: " + str(candle.low))
            simulation_log.append("close: " + str(candle.close))

            cost_per_share = candle.close * Simulation.cost_per_share_ratio
            simulation_log.append("raw cost per share: " + str(cost_per_share))
            if cost_per_share < Simulation.cost_per_share_min:
                simulation_log.append("cost per share is below minimum value")
                cost_per_share = Simulation.cost_per_share_min
                simulation_log.append("updated cost per share: " + str(cost_per_share))

            simulation_log.append("\n------------------------------------------")
            simulation_log.append("Predictions:")
            for key in candle.predictions:
                Prediction = candle.predictions[key]
                pstring = key + "=" + str(Prediction.value)
                simulation_log.append(pstring)

            avg_rise = get_avg_prediction_by_type(candle.predictions, PredictionType.rise)
            avg_cor = get_avg_prediction_by_type(candle.predictions, PredictionType.close_to_open_ratio)
            avg_hor = get_avg_prediction_by_type(candle.predictions, PredictionType.high_to_open_ratio)
            avg_lor = get_avg_prediction_by_type(candle.predictions, PredictionType.low_to_open_ratio)

            simulation_log.append("\naverages")
            simulation_log.append("avg predicted rise: " + str(avg_rise))
            simulation_log.append("avg predicted close to open ratio: " + str(avg_cor))
            simulation_log.append("avg predicted high to open ratio: " + str(avg_hor))
            simulation_log.append("avg predicted low to open ratio: " + str(avg_lor))

            # open new pending trades (LONG/SHORT)
            if (avg_rise > trade_config.min_rise_for_long) and (avg_cor > trade_config.min_cor_for_long):
                entry_price = candle.close * (1 + trade_config.entry_price_ratio)
                take_ratio = avg_cor * trade_config.take_profit_ratio
                take_price = entry_price * (1 + take_ratio)
                stop_ratio = take_ratio * trade_config.stop_loss_to_take_profit_ratio
                stop_price = candle.close * (1 - stop_ratio)

                new_trade = Trade(entry_price, stop_price, take_price, cost_per_share, Simulation.share_count_default)
                self.pending_trades.append(new_trade)

                simulation_log.append("     -------------------------------------------------------")
                simulation_log.append("     \nopened new trade: Long")
                simulation_log.append("     entry price: "+str(entry_price))
                simulation_log.append("     take price: " + str(take_price))
                simulation_log.append("     stop price: " + str(stop_price))
                simulation_log.append("     -------------------------------------------------------")

            elif (avg_rise < trade_config.max_rise_for_short) and (avg_cor < trade_config.max_cor_for_short):
                entry_price = candle.close * (1 - trade_config.entry_price_ratio)
                take_ratio = avg_cor * trade_config.take_profit_ratio
                take_price = entry_price * (1 - take_ratio)
                stop_ratio = take_ratio * trade_config.stop_loss_to_take_profit_ratio
                stop_price = entry_price * (1 + stop_ratio)

                new_trade = Trade(entry_price, stop_price, take_price, cost_per_share, Simulation.share_count_default)
                self.pending_trades.append(new_trade)

                simulation_log.append("     -------------------------------------------------------")
                simulation_log.append("     \nopened new trade: Short")
                simulation_log.append("     entry price: " + str(entry_price))
                simulation_log.append("     take price: " + str(take_price))
                simulation_log.append("     stop price: " + str(stop_price))
                simulation_log.append("     -------------------------------------------------------")

                print("opened Short trade")

            else:
                simulation_log.append("     avg rise is between thresholds - no LONG/SHORT trade to open")
                print("avg rise is between thresholds")

            # open peak pending trade
            hcr = avg_hor - avg_cor
            if hcr > trade_config.min_abs_hcr_for_peak_short:
                entry_price = candle.close * (1 + avg_hor)

                take_ratio = hcr * trade_config.take_profit_ratio
                take_price = entry_price * (1 - take_ratio)

                stop_ratio = take_ratio * trade_config.stop_loss_to_take_profit_ratio
                stop_price = entry_price * (1 + stop_ratio)

                new_trade = Trade(entry_price, stop_price, take_price, cost_per_share, Simulation.share_count_default)
                self.pending_trades.append(new_trade)

                simulation_log.append("     -------------------------------------------------------")
                simulation_log.append("     \nopened new trade: Peak")
                simulation_log.append("     entry price: " + str(entry_price))
                simulation_log.append("     take price: " + str(take_price))
                simulation_log.append("     stop price: " + str(stop_price))
                simulation_log.append("     -------------------------------------------------------")

                print("opened peak trade")

            # open low short trade
            lcr = avg_cor - avg_lor
            if lcr > trade_config.min_abs_lcr_for_low_short:
                entry_price = candle.close * (1 - trade_config.entry_price_ratio)

                take_ratio = lcr * trade_config.take_profit_ratio
                take_price = entry_price * (1 - take_ratio)

                stop_ratio = take_ratio * trade_config.stop_loss_to_take_profit_ratio
                stop_price = entry_price * (1 + stop_ratio)

                new_trade = Trade(entry_price, stop_price, take_price, cost_per_share, Simulation.share_count_default)
                self.pending_trades.append(new_trade)

                simulation_log.append("     -------------------------------------------------------")
                simulation_log.append("     \nopened new trade: trade open to low (low short trade)")
                simulation_log.append("     entry price: " + str(entry_price))
                simulation_log.append("     take price: " + str(take_price))
                simulation_log.append("     stop price: " + str(stop_price))
                simulation_log.append("     -------------------------------------------------------")

                print("open low short trade")

        with open('D:\shmekels\Data\simulation_logs\simulation_log_'+self.name+'.txt', 'w') as filehandle:
            for listitem in simulation_log:
                filehandle.write('%s\n' % listitem)
        print("done")


def get_avg_prediction_by_type(predictors={}, prediction_type=None):
    total = 0
    count = 0
    for key, FakePrediction in predictors.items():
        if FakePrediction.type == prediction_type:
            count = count + 1
            total = total + FakePrediction.value
            # print(key, '->', FakePrediction.value)
    avg_value = total / count
    return avg_value


def fake_predict_candles(candles):
    fake_ranges = [1, 2, 3]
    fake_predict_types = [PredictionType.rise,
                          PredictionType.close_to_open_ratio,
                          PredictionType.high_to_open_ratio,
                          PredictionType.low_to_open_ratio]

    def get_fake_value_by_type(fake_type):
        if fake_type == PredictionType.rise:
            return uniform(0, 1)
        if fake_type == PredictionType.high_to_open_ratio:
            return uniform(0, 0.1)
        if fake_type == PredictionType.close_to_open_ratio:
            return uniform(-0.1, 0.1)
        if fake_type == PredictionType.low_to_open_ratio:
            return uniform(-0.1, 0)

    for candle in candles:
        for range in fake_ranges:
            for type in fake_predict_types:
                fake_prediction = FakePrediction(type, range)
                fake_prediction.value = get_fake_value_by_type(type)
                fake_id = str(type) + "_" + str(range)
                candle.predictions[fake_id] = fake_prediction
                # print("fake value: "+str(fake_prediction.value))
    return candles


def load_by_exported_config():
    dir_name = Location.models_dir
    files = get_list_of_files(dir_name)
    for file in files:
        print(file)


def get_list_of_files(dir_name):
    file_list = os.listdir(dir_name)
    all_files = list()
    for entry in file_list:
        full_path = os.path.join(dir_name, entry)
        if os.path.isdir(full_path):
            all_files = all_files + get_list_of_files(full_path)
        else:
            all_files.append(full_path)

    return all_files
