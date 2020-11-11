from ibapi.contract import Contract
from Trader.Api.api_client import ApiClient
import time

# Historical data requires paying for TWS as explained in these links:
# http://interactivebrokers.github.io/tws-api/historical_data.html
# https://groups.io/g/twsapi/topic/no_market_data_permissions/24243137?p=,,,20,0,0,0::recentpostdate%2Fsticky,,,20,2,0,24243137


class HistoricalData(ApiClient):
    def __init__(self):
        ApiClient.__init__(self)
        self.historical_data_answered = False

    # ----- Interactive Brokers API callbacks: ----- #

    def historicalData(self, request_id, bar):
        print(f'Time: {bar.date} Close: {bar.close}')

    def historicalDataEnd(self, request_id: int, start: str, end: str):
        super().historicalDataEnd(request_id, start, end)
        print("HistoricalDataEnd. request_id:", request_id, "from", start, "to", end)
        self.historical_data_answered = True


app = HistoricalData()
if app.connect():
    # Create contract object
    apple_contract = Contract()
    apple_contract.symbol = 'AAPL'
    apple_contract.secType = 'STK'
    apple_contract.exchange = 'SMART'
    apple_contract.currency = 'USD'

    app.reqHistoricalData(1, apple_contract, '', '2 D', '1 hour', 'BID', 0, 2, False, [])

    while not app.historical_data_answered:
        print("sleeping")
        time.sleep(1)  # Sleep interval to allow time for incoming price data
    print("woke up")
    app.disconnect()