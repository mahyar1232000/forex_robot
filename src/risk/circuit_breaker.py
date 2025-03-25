"""
circuit_breaker.py

Implements circuit breaker logic to halt trading when drawdown exceeds a defined limit.
"""


def check_circuit_breaker(trade_history, current_balance, drawdown_limit=0.05):
    if not trade_history:
        return True
    equity_high = max(record['balance'] for record in trade_history)
    drawdown = (equity_high - current_balance) / equity_high
    return drawdown < drawdown_limit
