import MetaTrader5 as mt5
import datetime
import logging


class DataManager:
    def __init__(self, mode='live'):
        self.mode = mode
        self.logger = logging.getLogger(__name__)
        if not mt5.initialize():
            raise RuntimeError("Failed to initialize MT5 in DataManager.")
        self.symbol = "EURUSD"
        if not mt5.symbol_select(self.symbol, True):
            raise RuntimeError(f"Failed to select symbol: {self.symbol}")

    def get_historical_data(self):
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=5)
        rates = mt5.copy_rates_range(self.symbol, mt5.TIMEFRAME_H1, start, end)
        if rates is None:
            self.logger.warning("No historical data returned.")
            return []
        return rates

    def get_latest_price(self):
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return None
        return tick.bid

    def get_market_data(self):
        tick = mt5.symbol_info_tick(self.symbol)
        return {
            'bid': tick.bid if tick else None,
            'ask': tick.ask if tick else None,
            'time': tick.time if tick else None
        }
