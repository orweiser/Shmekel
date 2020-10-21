from Trader.Core.order import *


class TradeConfig:
    min_rise_for_long = 0.75
    max_rise_for_short = 0.25
    min_cor_for_long = 0.025
    max_cor_for_short = -0.025
    min_abs_hcr_for_peak_short = 0.025
    min_abs_lcr_for_low_short = 0.025
    entry_price_ratio = 0.01
    take_profit_ratio = 0.75
    stop_loss_to_take_profit_ratio = 0.25

    def LogPrint(self):
        output = "\nTrade Config:\n"
        output = output+"\nMin rise for long: "+str(self.min_rise_for_long)
        output = output+"\nMax rise for short: "+str(self.max_rise_for_short)
        output = output+"\nMin close to open ratio for long: "+str(self.min_cor_for_long)
        output = output+"\nMax close to open ratio for Short: " + str(self.max_cor_for_short)
        output = output+"\nMin (abs) high to close ratio for peak short: " + str(self.min_abs_hcr_for_peak_short)
        output = output+"\nMin (abs) low to close ratio for short low: " + str(self.min_abs_lcr_for_low_short)
        output = output+"\nEntry price ratio: " + str(self.entry_price_ratio)
        output = output+"\nTake profit ratio: " + str(self.take_profit_ratio)
        output = output+"\nStop loss to take profit ratio: " + str(self.stop_loss_to_take_profit_ratio)
        return output

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
        self.orders = []