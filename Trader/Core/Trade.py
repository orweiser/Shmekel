from Trader.Core.order import *


class TradeConfig:
    min_rise_for_long = 0.75
    max_rise_for_short = 0.25
    min_cor_for_long = 0.75
    max_cor_for_short = -0.75
    min_abs_hcr_for_peak_short = 0.5
    min_abs_lcr_for_low_short = 0.5


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
