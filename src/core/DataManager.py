"""
DataManager.py
==============
Manages retrieval, storage, and processing of both historical and real-time trading data.
"""

import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)


class DataManager:
    def __init__(self):
        self.historical_path = os.path.join("data", "historical.csv")

    def load_historical_data(self):
        try:
            if not os.path.exists(self.historical_path):
                raise FileNotFoundError(f"Historical data file not found: {self.historical_path}")

            data = pd.read_csv(self.historical_path)

            # Handle both 'date' and 'time' column names
            if "date" in data.columns:
                data.rename(columns={"date": "time"}, inplace=True)
            elif "time" not in data.columns:
                raise ValueError(f"Missing timestamp column. Available columns: {list(data.columns)}")

            # Convert timestamp to datetime format
            data["time"] = pd.to_datetime(data["time"], errors="coerce")
            data.dropna(subset=["time"], inplace=True)

            logger.info(f"Successfully loaded historical data with {len(data)} rows.")
            return data
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            raise
