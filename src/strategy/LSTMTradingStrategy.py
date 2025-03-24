"""
LSTMTradingStrategy.py (strategies)
===================================
Provides an alternative implementation of the LSTM-based trading strategy.
(Optional: Add alternative strategy code here if desired.)
"""

# (Implementation code here if desired)

from src.core.LSTMTradingStrategy import LSTMTradingStrategy as CoreLSTMTradingStrategy


class LSTMTradingStrategy(CoreLSTMTradingStrategy):
    def __init__(self, model_path):
        super().__init__(model_path)
