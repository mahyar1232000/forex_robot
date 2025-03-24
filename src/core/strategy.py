"""
strategy.py
===========
Contains primary trading strategy functions.
- LSTMTradingStrategy: Trains (or loads) the LSTM model for predictions.
- get_data: Retrieves historical data via MT5 (used in live mode).
- generate_signal: Generates a trading signal based on technical indicators.
"""


class Strategy:
    def __init__(self):
        self.threshold = 1.1050

    def generate_signal(self, market_data):
        price = market_data.get("close", None) if isinstance(market_data, dict) else market_data.close
        if price is None:
            return None
        if price > self.threshold:
            return "BUY"
        else:
            return "SELL"
