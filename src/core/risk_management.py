from config.settings import RISK_PERCENT, STOP_LOSS_PIPS, LOT_SIZE


def calculate_lot_size(account_balance):
    """Calculate lot size based on risk percentage."""
    risk_amount = account_balance * (RISK_PERCENT / 100)
    lot_size = risk_amount / (STOP_LOSS_PIPS * 10)  # Assuming 1 pip = $10 for a standard lot
    return round(lot_size, 2)


def calculate_stop_loss_take_profit(order_type, current_price):
    """Calculate stop-loss and take-profit levels."""
    if order_type == "BUY":
        stop_loss = current_price - STOP_LOSS_PIPS * 0.0001
        take_profit = current_price + TAKE_PROFIT_PIPS * 0.0001
    else:
        stop_loss = current_price + STOP_LOSS_PIPS * 0.0001
        take_profit = current_price - TAKE_PROFIT_PIPS * 0.0001
    return stop_loss, take_profit
