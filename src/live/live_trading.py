from src.core.strategy import get_data, generate_signal
from src.core.broker import execute_trade
from src.core.risk_management import calculate_lot_size, calculate_stop_loss_take_profit
from src.utils.logger import log_message


def run_live_trading():
    """Run live trading."""
    data = get_data(mt5.TIMEFRAME_M1, 100)  # Example: 1-minute timeframe, 100 candles
    signal = generate_signal(data)
    if signal:
        lot_size = calculate_lot_size(mt5.account_info().balance)
        stop_loss, take_profit = calculate_stop_loss_take_profit(signal, data.iloc[-1]['close'])
        execute_trade(signal, lot_size, stop_loss, take_profit)
        log_message(f"Live trade executed: {signal} trade")
