import unittest
import pandas as pd
from forex_robot.strategy import generate_signal

class TestStrategy(unittest.TestCase):

    def test_generate_signal(self):
        data = pd.DataFrame({
            'open': [1.2000],
            'close': [1.2050]
        })
        signal = generate_signal(data)
        self.assertEqual(signal, "BUY", "Signal generation failed.")

if __name__ == '__main__':
    unittest.main()
