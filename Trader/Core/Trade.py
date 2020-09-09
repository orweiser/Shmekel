from Trader.Core.order import *


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




