"""
AdvancedRiskManager.py

Provides dynamic risk management functions.
"""


def calculate_dynamic_risk(account_balance, base_risk_percent):
    INITIAL_BALANCE = 200.0
    if account_balance >= INITIAL_BALANCE:
        factor = 1 + min((account_balance - INITIAL_BALANCE) / INITIAL_BALANCE * 0.1, 1.0)
    else:
        factor = max(0.5, 1 - (INITIAL_BALANCE - account_balance) / INITIAL_BALANCE * 0.1)
    return base_risk_percent * factor
