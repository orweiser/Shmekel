

class Simulation:
    initial_funds = 100000


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

class Fake_Prediction:
    def __init__(self, prediction_type, rise_in_k_candle):
        self.type = prediction_type
        self.value = {}
        self.ks = []