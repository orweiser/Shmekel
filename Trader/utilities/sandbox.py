class Prediction:
    def __init__(self, rise_in_k_candle):
        self.rise = {}
        self.ks = []
        for k, rise in rise_in_k_candle:
            self.rise[k] = rise
            self.ks.append(k)


class Candle:
    def __init__(self, datetime, open, high, low, close, volume, prediction=None, previous=None):
        self.datetime = datetime
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.prediction = prediction
        self.previous = previous
        self.o2h = (self.high - self.open) / self.open
        self.o2c = (self.close - self.open) / self.open
        self.o2l = (self.low - self.open) / self.open
        if previous:
            self.prev_close_2_open = (self.open - self.previous.close) / self.previous.close
        else:
            self.prev_close_2_open = 0

    def set_previous(self, previous):
        self.previous = previous
        self.prev_close_2_open = (self.open - self.previous.close) / self.previous.close

    # def __repr__(self):
    #     return ""


class Asset:
    def __init__(self, name, asset_type, sector, candles=[]):
        self.name = name
        self.type = asset_type
        self.sector = sector
        self.candles = candles


class Trade:
    def __init__(self, entry_price, stop_loss, take_profit, cost_per_share, shares=0):
        self.status = "pending"
        self.shares = shares
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.no_action_counter = 0
        self.no_action_time = 0
        self.profit_loss = 0
        self.cost_per_share = cost_per_share


