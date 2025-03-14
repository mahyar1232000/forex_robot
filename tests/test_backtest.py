import unittest
from forex_robot.backtest import run_backtest  # Assuming you have a run_backtest function

class TestBacktest(unittest.TestCase):

    def test_run_backtest(self):
        result = run_backtest()
        self.assertIsNotNone(result, "Backtest run failed.")

if __name__ == '__main__':
    unittest.main()
