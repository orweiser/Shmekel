from ibapi.client import MarketDataTypeEnum
from ibapi.contract import Contract
from Trader.Api.api_client import ApiClient
import time


class MarketAndAccountData(ApiClient):
    def __init__(self):
        ApiClient.__init__(self)
        self.market_data_answered = False
        self.account_data_received = False

    # this should probably be part of the basic ApiClient connection process
    def get_account_data(self):
        self.reqAccountUpdates(True, self.account_code)
        while not app.account_data_received:
            time.sleep(1)  # Sleep interval to allow time for incoming price data
        self.reqAccountUpdates(False, self.account_code)
        # should return desired account data (collected in the callbacks)

    # ----- Interactive Brokers API callbacks: ----- #

    def updateAccountValue(self, key: str, val: str, currency: str, account_name: str):
        super().updateAccountValue(key, val, currency, account_name)
        print("UpdateAccountValue. Key:", key, "Value:", val,
              "Currency:", currency, "AccountName:", account_name)

    def updatePortfolio(self, contract: Contract, position: float,
                        market_price: float, market_value: float,
                        average_cost: float, unrealized_pnl: float,
                        realized_pnl: float, account_name: str):
        super().updatePortfolio(contract, position, market_price, market_value,
                                average_cost, unrealized_pnl, realized_pnl, account_name)
        print("UpdatePortfolio.", "Symbol:", contract.symbol, "SecType:", contract.secType, "Exchange:",
              contract.exchange, "Position:", position, "MarketPrice:", market_price,
              "MarketValue:", market_value, "AverageCost:", average_cost,
              "UnrealizedPNL:", unrealized_pnl, "RealizedPNL:", realized_pnl,
              "AccountName:", account_name)

    def updateAccountTime(self, time_stamp: str):
        super().updateAccountTime(time_stamp)
        print("UpdateAccountTime. Time:", time_stamp)

    def accountDownloadEnd(self, account_name: str):
        super().accountDownloadEnd(account_name)
        print("AccountDownloadEnd. Account:", account_name)
        self.account_data_received = True

    def tickPrice(self, request_id, tick_type, price, attrib):
        if tick_type == 67 and request_id == 1:
            print('The current ask price is: ', price)
            self.market_data_answered = True


app = MarketAndAccountData()
if app.connect():
    app.get_account_data()
    # Create contract object
    apple_contract = Contract()
    apple_contract.symbol = 'AAPL'
    apple_contract.secType = 'STK'
    apple_contract.exchange = 'SMART'
    apple_contract.currency = 'USD'

    app.reqMarketDataType(MarketDataTypeEnum.DELAYED)
    # Request Market Data
    app.reqMktData(1, apple_contract, '', False, False, [])

    while not app.market_data_answered:
        print("sleeping")
        time.sleep(1)  # Sleep interval to allow time for incoming price data
    print("woke up")
    app.disconnect()
