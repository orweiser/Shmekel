class Simulation:
    initial_funds = 100000
    cost_per_share_ratio = 0.001
    cost_per_share_min = 0.05
    share_count_default = 100

class Location:
    models_dir = "D:\\shmekels\\downloads\\models"
    stocks_root = "D:\\shmekels\\Data\\Stocks_by_sector"


class AssetType:
    stock = "stock"
    index = "index"
    currency = "currency"


class RunMode:
    simulate = "simulate"
    demo = "demo"
    real_time = "real_time"


class OrderStatus:
    pending = "pending"
    running = "running"
    complete = "complete"
    cancelled = "cancelled"


class OrderType:
    limit = "LMT"
    limit_on_close = "LOC"
    stop = "STP"


class PredictionType:
    rise = "rise"
    high_to_open_ratio = "hor"
    close_to_open_ratio = "cor"
    low_to_open_ratio = "lor"


class FakePrediction:
    def __init__(self, prediction_type, rise_in_k_candle):
        self.type = prediction_type
        self.value = {}
        self.ks = []
