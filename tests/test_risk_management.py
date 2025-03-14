import unittest
from forex_robot.risk_management import calculate_lot_size, calculate_stop_loss_take_profit

class TestRiskManagement(unittest.TestCase):

    def test_calculate_lot_size(self):
        account_balance = 10000
        lot_size = calculate_lot_size(account_balance)
        self.assertGreater(lot_size, 0, "Lot size calculation failed.")

    def test_calculate_stop_loss_take_profit(self):
        current_price = 1.2000
        stop_loss, take_profit = calculate_stop_loss_take_profit("BUY", current_price)
        self.assertIsNotNone(stop_loss, "Stop loss calculation failed.")
        self.assertIsNotNone(take_profit, "Take profit calculation failed.")

if __name__ == '__main__':
    unittest.main()
