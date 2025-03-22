import unittest
from src.core.strategy import Strategy


class TestStrategy(unittest.TestCase):
    def test_generate_signal(self):
        strategy = Strategy()
        market_data = {"close": 1.11}
        signal = strategy.generate_signal(market_data)
        self.assertIn(signal, ["BUY", "SELL"])


if __name__ == '__main__':
    unittest.main()
