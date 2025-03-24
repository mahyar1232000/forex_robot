"""
train.py (modes)
================
Launcher module for training mode.
Trains the LSTM model using historical data.
"""

"""
train.py

Mode launcher for model training.
"""

from src.core.LSTMTradingStrategy import LSTMTradingStrategy
import sys

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python train.py <data_path> <model_path>")
    else:
        data_path = sys.argv[1]
        model_path = sys.argv[2]
        strategy = LSTMTradingStrategy(data_path, model_path)
        summary = strategy.execute_strategy()
        print("Model training completed. Summary:")
        print(summary)
