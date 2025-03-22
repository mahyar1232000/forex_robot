import unittest
from src.core.DataManager import DataManager
import pandas as pd
from io import StringIO


class TestDataManager(unittest.TestCase):
    def setUp(self):
        # Create a dummy CSV data
        self.csv_data = "Date,close\n2020-01-01,1.1\n2020-01-02,1.2"
        self.data_source = StringIO(self.csv_data)
        self.manager = DataManager(data_source=self.data_source)

    def test_load_data(self):
        data = self.manager.load_data()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIn('close', data.columns)


if __name__ == '__main__':
    unittest.main()
