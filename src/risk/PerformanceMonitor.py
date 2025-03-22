"""
PerformanceMonitor.py

Monitors and reports trading performance metrics.
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PerformanceMonitor:
    def __init__(self):
        self.balance_history = []

    def update(self, current_balance):
        self.balance_history.append(current_balance)

    def report(self):
        if not self.balance_history:
            logger.info("No performance data available.")
            return
        final_balance = self.balance_history[-1]
        max_balance = max(self.balance_history)
        drawdown = (max_balance - final_balance) / max_balance if max_balance > 0 else 0
        logger.info(
            f"Performance Report: Final Balance: {final_balance:.2f}, Max Balance: {max_balance:.2f}, Drawdown: {drawdown * 100:.2f}%")
