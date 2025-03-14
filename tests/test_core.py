import unittest
import numpy as np
import pandas as pd
from src.core.risk import AdvancedRiskManager
from src.core.circuit_breaker import TradingCircuitBreaker


class TestTradingSystem(unittest.TestCase):
    def test_risk_calculation(self):
        data = pd.DataFrame({'high': [1.1, 1.2], 'low': [1.0, 1.1], 'close': [1.05, 1.15]})
        rm = AdvancedRiskManager()
        atr = rm.calculate_atr(data)
        self.assertAlmostEqual(atr, 0.075, delta=0.01)

    def test_circuit_breaker(self):
        cb = TradingCircuitBreaker(max_drawdown=0.1)
        cb.check_balance(10000)
        with self.assertRaises(TradingHaltedError):
            cb.check_balance(8500)


if __name__ == "__main__":
    unittest.main()
