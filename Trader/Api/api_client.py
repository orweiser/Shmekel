from ibapi.client import EClient, MarketDataTypeEnum
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.ticktype import TickTypeEnum

import threading
import time


class ApiClient(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.order_id = 0
        self.account_code = ''

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
                return True
            time.sleep(1)  # Yuck... need to find better solution than sleeping...
        print("Failed to connect")
        return False

    # ----- Interactive Brokers API callbacks: ----- #

    def nextValidId(self, order_id: int):
        self.order_id = order_id

    def managedAccounts(self, accounts_list: str):
        super().managedAccounts(accounts_list)
        print("Account list:", accounts_list)
        self.account_code = accounts_list
