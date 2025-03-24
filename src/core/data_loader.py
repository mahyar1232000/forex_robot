"""
data_loader.py
==============
Provides functions to load and preprocess historical data from CSV files.
"""

import pandas as pd
from os import path


def load_csv_data(filename):
    filepath = path.join("data", filename)
    try:
        data = pd.read_csv(filepath)
        return data
    except Exception as e:
        raise Exception(f"Error loading {filename}: {e}")
