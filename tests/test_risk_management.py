import unittest
from src.core.risk_management import calculate_lot_size


class TestRiskManagement(unittest.TestCase):
    def test_calculate_lot_size(self):
        lot_size = calculate_lot_size(10000, 0.02, 0.05)
        self.assertGreater(lot_size, 0)


if __name__ == '__main__':
    unittest.main()
