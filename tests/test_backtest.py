import unittest
from src.backtest.backtest import run_backtest


class TestBacktest(unittest.TestCase):
    def test_run_backtest(self):
        results = run_backtest()
        self.assertIn("final_balance", results)


if __name__ == '__main__':
    unittest.main()
