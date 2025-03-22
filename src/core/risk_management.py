"""
risk_management.py
==================
Defines functions to calculate lot size, stop loss, and take profit.
"""


def calculate_lot_size(account_balance, risk_percentage, stop_loss_distance):
    risk_amount = account_balance * risk_percentage
    if stop_loss_distance == 0:
        return 0
    lot_size = risk_amount / stop_loss_distance
    return lot_size
