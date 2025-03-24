import unittest
from src.core.broker import MT5Broker


class TestBroker(unittest.TestCase):
    def test_connection(self):
        broker = MT5Broker()
        broker.connect()
        self.assertTrue(broker.connected)


if __name__ == '__main__':
    unittest.main()
