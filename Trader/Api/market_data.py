from ibapi.client import MarketDataTypeEnum
from ibapi.contract import Contract
from Trader.Api.api_client import ApiClient
import time


class MarketData(ApiClient):
    def __init__(self):
        ApiClient.__init__(self)
        self.market_data_answered = False

    def tickPrice(self, request_id, tick_type, price, attrib):
        if tick_type == 67 and request_id == 1:
            print('The current ask price is: ', price)
            self.market_data_answered = True


app = MarketData()
if app.connect():
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
