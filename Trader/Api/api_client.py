from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.execution import Execution, ExecutionFilter
import threading
import time


class ApiClient(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.order_id = 0
        self.account_code = ''
        self.account_data_received = False
        self.account_info = {
            "available_funds": 0,
            "buying_power": 0,
            "initial_margin": 0,
            "maintenance_margin": 0
        }
        self.portfolio = []
        self.trade_log_received = False

    def connect(self, host='127.0.0.1', port=7497, client_id=123):
        if self.order_id > 0:
            print("Already connected")
            return True
        super().connect(host, port, client_id)
        thread = threading.Thread(target=super().run, daemon=True)
        thread.start()
        for tries in range(5):
            print("Connecting...")
            if self.order_id > 0:
                self.get_account_data()
                return True
            time.sleep(1)  # Yuck... need to find better solution than sleeping...
        print("Failed to connect")
        return False

    def get_account_data(self):
        self.reqAccountUpdates(True, self.account_code)
        while not self.account_data_received:
            time.sleep(1)  # Sleep interval to allow time for incoming account data
        self.reqAccountUpdates(False, self.account_code)
        self.account_data_received = False
        return {"account_info": self.account_info, "portfolio": self.portfolio}

    def get_trade_log(self, request_id):
        self.reqExecutions(request_id, ExecutionFilter())
        while not self.trade_log_received:
            time.sleep(1)  # Sleep interval to allow time for incoming account data

    # ----- Interactive Brokers API callbacks: ----- #

    def nextValidId(self, order_id: int):
        self.order_id = order_id

    def managedAccounts(self, accounts_list: str):
        super().managedAccounts(accounts_list)
        print("Account list:", accounts_list)
        self.account_code = accounts_list

    def updateAccountValue(self, key: str, val: str, currency: str, account_name: str):
        super().updateAccountValue(key, val, currency, account_name)
        print("UpdateAccountValue. Key:", key, "Value:", val,
              "Currency:", currency, "AccountName:", account_name)
        if key is "AvailableFunds":
            self.account_info["available_funds"] = int(val)
        elif key is "BuyingPower":
            self.account_info["buying_power"] = int(val)
        elif key is "InitMarginReq":
            self.account_info["initial_margin"] = int(val)
        elif key is "MaintMarginReq":
            self.account_info["maintenance_margin"] = int(val)

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
        self.portfolio.append({
            "symbol": contract.symbol,
            "sec_type": contract.secType,
            "position": position,
            "market_price": market_price,
            "market_value": market_value,
            "average_cost": average_cost,
            "realized_pnl": realized_pnl
        })

    def updateAccountTime(self, time_stamp: str):
        super().updateAccountTime(time_stamp)
        print("UpdateAccountTime. Time:", time_stamp)

    def accountDownloadEnd(self, account_name: str):
        super().accountDownloadEnd(account_name)
        print("AccountDownloadEnd. Account:", account_name)
        self.account_data_received = True

    def execDetails(self, request_id: int, contract: Contract, execution: Execution):
        super().execDetails(request_id, contract, execution)
        print("execDetails.", "request_id:", request_id, "Symbol:", contract.symbol, "SecType:", contract.secType, "Exchange:",
              contract.exchange, "execution:", execution)
        self.trade_log_received = True
