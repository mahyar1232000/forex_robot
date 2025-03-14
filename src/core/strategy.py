import pandas as pd
from config.settings import SYMBOL


def get_data(timeframe, num_candles):
    """Retrieve historical data."""
    rates = mt5.copy_rates_from_pos(SYMBOL, timeframe, 0, num_candles)
    return pd.DataFrame(rates)


def generate_signal(data):
    """Generate buy/sell signals."""
    last_candle = data.iloc[-1]
    if last_candle['close'] > last_candle['open']:
        return "BUY"
    elif last_candle['close'] < last_candle['open']:
        return "SELL"
    return None
