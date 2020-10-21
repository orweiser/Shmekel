from Trader.Core.enums import OrderStatus, OrderType, RunMode
from Trader.Core.Candle import *


class Order:
    def __init__(self, order_type):
        self.type = order_type
        self.status = OrderStatus.pending

    def CheckStatus(self,candle):
        raise Exception("no implementation for Order->Run")

    def run(self):
        #self.status = OrderStatus.running
        raise Exception("no implementation for Order->Run")
