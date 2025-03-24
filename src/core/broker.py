"""
broker.py

Manages live broker connections and trade execution via MetaTrader5.
Credentials are loaded from config/secret.yaml.
"""

import MetaTrader5 as mt5
from typing import Dict, Optional, List
from src.utils.config import load_config
from src.utils.logger import get_logger
from config.settings import SYMBOL, MAGIC_NUMBER

logger = get_logger(__name__)


class MT5Broker:
    """Enhanced MT5 trading broker implementation"""

    def __init__(self):
        self.connected = False
        self.config = load_config('secret.yaml')
        self.account_info: Optional[Dict] = None
        self.symbol = SYMBOL
        self.magic = MAGIC_NUMBER

    def connect(self) -> bool:
        try:
            if not mt5.initialize():
                logger.error("MT5 initialization failed")
                return False
            authorized = mt5.login(
                login=self.config['mt5']['account'],
                password=self.config['mt5']['password'],
                server=self.config['mt5']['server']
            )
            if authorized:
                self.connected = True
                self._refresh_account_info()
                logger.info(f"Connected to {self.config['mt5']['account']}@{self.config['mt5']['server']}")
                return True
            else:
                logger.error("MT5 login failed")
                return False
        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
            return False

    def _refresh_account_info(self):
        try:
            info = mt5.account_info()
            if info:
                self.account_info = {
                    'balance': info.balance,
                    'equity': info.equity,
                    'margin': info.margin,
                    'free_margin': info.margin_free
                }
        except Exception as e:
            logger.error(f"Failed to refresh account info: {str(e)}")

    def get_account_info(self) -> Dict:
        self._refresh_account_info()
        return self.account_info if self.account_info else {}

    def get_open_positions(self) -> List[Dict]:
        positions = mt5.positions_get()
        return [p._asdict() for p in positions] if positions else []

    def get_trade_history(self) -> List[Dict]:
        from datetime import datetime
        deals = mt5.history_deals_get(datetime(2000, 1, 1), datetime.now())
        return [d._asdict() for d in deals] if deals else []

    def execute_trade(self, order_type: int, lot_size: float,
                      stop_loss: float, take_profit: float) -> bool:
        if not self.connected:
            if not self.connect():
                return False
        try:
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                logger.error(f"Symbol {self.symbol} not found")
                return False

            # Ensure the symbol is visible
            if not symbol_info.visible:
                if not mt5.symbol_select(self.symbol, True):
                    logger.error(f"Failed to select symbol {self.symbol}")
                    return False

            # Adjust the volume (lot size) as needed
            min_volume = symbol_info.volume_min
            volume_step = symbol_info.volume_step
            if lot_size < min_volume:
                logger.warning(
                    f"Provided lot_size {lot_size} is below the minimum allowed ({min_volume}). Adjusting to minimum volume.")
                lot_size = min_volume
            else:
                remainder = lot_size % volume_step
                if remainder != 0:
                    adjusted_volume = lot_size - remainder
                    if adjusted_volume < min_volume:
                        adjusted_volume = min_volume
                    logger.info(
                        f"Adjusted lot_size from {lot_size} to {adjusted_volume} to meet volume step requirement ({volume_step}).")
                    lot_size = adjusted_volume

            current_bid = symbol_info.bid
            current_ask = symbol_info.ask

            # Define a minimum stop distance (e.g. 10 points)
            MIN_STOP_DISTANCE = symbol_info.point * 10

            # Validate stops for SELL orders
            if order_type == mt5.ORDER_TYPE_SELL:
                if stop_loss <= current_bid:
                    logger.error(
                        f"For SELL orders, stop loss ({stop_loss}) must be above the current bid ({current_bid})")
                    return False
                if (stop_loss - current_bid) < MIN_STOP_DISTANCE:
                    logger.error(
                        f"For SELL orders, stop loss ({stop_loss}) is too close to the current bid ({current_bid}). Minimum distance is {MIN_STOP_DISTANCE}.")
                    return False
                if take_profit >= current_bid:
                    logger.error(
                        f"For SELL orders, take profit ({take_profit}) must be below the current bid ({current_bid})")
                    return False
                if (current_bid - take_profit) < MIN_STOP_DISTANCE:
                    logger.error(
                        f"For SELL orders, take profit ({take_profit}) is too close to the current bid ({current_bid}). Minimum distance is {MIN_STOP_DISTANCE}.")
                    return False

            # Validate stops for BUY orders
            elif order_type == mt5.ORDER_TYPE_BUY:
                if stop_loss >= current_ask:
                    logger.error(
                        f"For BUY orders, stop loss ({stop_loss}) must be below the current ask ({current_ask})")
                    return False
                if (current_ask - stop_loss) < MIN_STOP_DISTANCE:
                    logger.error(
                        f"For BUY orders, stop loss ({stop_loss}) is too close to the current ask ({current_ask}). Minimum distance is {MIN_STOP_DISTANCE}.")
                    return False
                if take_profit <= current_ask:
                    logger.error(
                        f"For BUY orders, take profit ({take_profit}) must be above the current ask ({current_ask})")
                    return False
                if (take_profit - current_ask) < MIN_STOP_DISTANCE:
                    logger.error(
                        f"For BUY orders, take profit ({take_profit}) is too close to the current ask ({current_ask}). Minimum distance is {MIN_STOP_DISTANCE}.")
                    return False

            # Round stops to match symbol precision
            sl_rounded = round(stop_loss, symbol_info.digits)
            tp_rounded = round(take_profit, symbol_info.digits)
            price = current_ask if order_type == mt5.ORDER_TYPE_BUY else current_bid

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": lot_size,
                "type": order_type,
                "price": price,
                "sl": sl_rounded,
                "tp": tp_rounded,
                "deviation": 20,
                "magic": self.magic,
                "comment": "MT5Broker Trade"
            }

            # Log the order request for debugging
            logger.info(f"Sending order request: {request}")

            result = mt5.order_send(request)
            if result is None:
                logger.error(
                    f"mt5.order_send returned None. MT5 last error: {mt5.last_error()}. Check order parameters and connectivity.")
                return False

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Trade executed successfully: {result}")
                return True
            else:
                logger.error(f"Trade execution failed: {result.comment}")
                return False

        except Exception as e:
            logger.error(f"Trade execution error: {str(e)}")
            return False


def initialize_broker() -> MT5Broker:
    broker = MT5Broker()
    if broker.connect():
        return broker
    raise ConnectionError("Failed to connect to MT5")
