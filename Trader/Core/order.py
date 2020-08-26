from Trader.Core.enums import OrderStatus, OrderType, RunMode


class Order:

    def __init__(self,type):
        self.type = type
        self.status = OrderStatus.pending

    def Run(self):
        self.status = OrderStatus.running
        raise Exception("no implementation for Order->Run")
