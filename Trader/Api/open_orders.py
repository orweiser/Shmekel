from ibapi.order import Order
from ibapi.order_state import OrderState
from ibapi.contract import Contract
from Trader.Api.api_client import ApiClient
import time

# Only active orders data can be retrieved as stated here:
# https://interactivebrokers.github.io/tws-api/open_orders.html


class OpenOrders(ApiClient):
    def __init__(self):
        ApiClient.__init__(self)
        self.open_orders_running = False
        self.current_orders = {}
        self.num_of_orders = 0

    def get_active_orders(self, force_refresh: bool = False):
        if not bool(self.current_orders) or force_refresh:
            if not self.open_orders_running:
                self.reqAllOpenOrders()
                self.open_orders_running = True
                self.num_of_orders = 0
            while self.open_orders_running:
                time.sleep(1)  # Sleep interval to allow time for incoming orders data
        return self.current_orders

    # ----- Interactive Brokers API callbacks: ----- #

    def openOrder(self, order_id: int, contract: Contract, order: Order, order_state: OrderState):
        super().openOrder(order_id, contract, order, order_state)
        print("OpenOrder. PermId: ", order.permId, "ClientId:", order.clientId, " OrderId:", order_id,
              "Account:", order.account, "Symbol:", contract.symbol, "SecType:", contract.secType,
              "Exchange:", contract.exchange, "Action:", order.action, "OrderType:", order.orderType,
              "TotalQty:", order.totalQuantity, "CashQty:", order.cashQty,
              "LmtPrice:", order.lmtPrice, "AuxPrice:", order.auxPrice, "Status:", order_state.status)
        order.contract = contract
        self.current_orders[contract.symbol][order_id] = order
        self.num_of_orders += 1

    def orderStatus(self, order_id: int, status: str, filled: float, remaining: float, avg_fill_price: float,
                    perm_id: int, parent_id: int, last_fill_price: float, client_id: int, why_held: str,
                    mkt_cap_price: float):
        super().orderStatus(order_id, status, filled, remaining, avg_fill_price, perm_id, parent_id, last_fill_price,
                            client_id, why_held, mkt_cap_price)
        print("OrderStatus. Id:", order_id, "Status:", status, "Filled:", filled,
              "Remaining:", remaining, "AvgFillPrice:", avg_fill_price,
              "PermId:", perm_id, "ParentId:", parent_id, "LastFillPrice:",
              last_fill_price, "ClientId:", client_id, "WhyHeld:",
              why_held, "MktCapPrice:", mkt_cap_price)

    def openOrderEnd(self):
        super().openOrderEnd()
        print("OpenOrderEnd")
        print("Received", self.num_of_orders, "orders")
        self.open_orders_running = False


app = OpenOrders()
if app.connect():
    orders = app.get_active_orders()
    print("orders:", orders)
    app.disconnect()
