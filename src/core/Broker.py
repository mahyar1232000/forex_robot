import MetaTrader5 as mt5
from config.secret import USERNAME, PASSWORD, SERVER
from config.settings import SYMBOL, MAGIC_NUMBER


class Broker:
    def __init__(self):
        pass

    def initialize_broker(self):
        """Initialize the broker connection using username and password."""
        if not mt5.initialize():
            print("Failed to initialize MT5. Check server and credentials.")
            return False

        # Login to the account
        authorized = mt5.login(USERNAME, password=PASSWORD, server=SERVER)
        if not authorized:
            print(f"Failed to login to account {USERNAME}. Check username, password, and server.")
            return False

        print(f"Logged in to account {USERNAME} on server {SERVER}")
        return True

    def execute_trade(order_type, lot_size, stop_loss, take_profit):
        """Execute a trade."""
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": lot_size,
            "type": order_type,
            "price": mt5.symbol_info_tick(SYMBOL).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(
                SYMBOL).bid,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 10,
            "magic": MAGIC_NUMBER,
            "comment": "Python script open",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Trade execution failed: {result.comment}")
            return False
        print(f"Trade executed successfully: {result.order}")
        return True
