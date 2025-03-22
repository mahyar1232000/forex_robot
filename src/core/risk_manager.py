# src/core/risk_manager.py
"""
risk_manager.py
===============
Provides additional risk management routines to complement basic risk functions.
"""

from src.core.risk_management import calculate_lot_size


class RiskManager:
    def __init__(self, account_balance, risk_percentage):
        self.account_balance = account_balance
        self.risk_percentage = risk_percentage

    def get_lot_size(self, stop_loss_distance):
        return calculate_lot_size(self.account_balance, self.risk_percentage, stop_loss_distance)
