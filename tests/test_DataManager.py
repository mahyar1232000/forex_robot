# tests/test_data_manager.py
import unittest
from src.core.DataManager import DataManager
import pandas as pd
import tempfile


class TestDataManager(unittest.TestCase):
    def setUp(self):
        self.csv_data = "Date,close\n2020-01-01,1.1\n2020-01-02,1.2"
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.csv')
        self.temp_file.write(self.csv_data)
        self.temp_file.close()
        self.manager = DataManager(data_source=self.temp_file.name)

    def tearDown(self):
        import os
        os.remove(self.temp_file.name)

    def test_load_data(self):
        data = self.manager.load_data()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIn('close', data.columns)


if __name__ == '__main__':
    unittest.main()
