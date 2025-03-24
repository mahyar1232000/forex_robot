"""
DataManager.py

Provides a simple interface to load historical data from a CSV file.
"""

import pandas as pd
from config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_historical_data():
    path = settings.historical_data_path
    try:
        data = pd.read_csv(path)
        logger.info(f"Loaded historical data from {path}, records: {len(data)}")
        return data
    except Exception as e:
        logger.error(f"Error loading historical data: {str(e)}")
        return None
