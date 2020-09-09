def initialize_predictors():
    print("initialize predictors")


class Asset:
    def __init__(self, name, asset_type, sector, candles=[]):
        self.name = name
        self.type = asset_type
        self.sector = sector
        self.candles = candles
        self.predictors = []
