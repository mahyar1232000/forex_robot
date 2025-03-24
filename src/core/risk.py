"""
risk.py
=======
Provides basic risk calculation functions for the trading strategy.
"""


def calculate_stop_loss(entry_price, risk_percentage, account_balance):
    risk_amount = account_balance * risk_percentage
    stop_loss = entry_price - risk_amount
    return stop_loss
