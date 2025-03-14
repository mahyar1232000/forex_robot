import numpy as np
import os
import logging
from tensorflow.keras.models import load_model
from src.train import train_model  # Import the training function


class LSTMTradingStrategy:
    def execute_strategy(self, data):
        self.logger = logging.getLogger(__name__)
        self.params = data
        model_path = data.get('model_path')
        # Check if the model exists; if not, train it.
        if not os.path.exists(model_path):
            self.logger.info("Trained LSTM model not found. Training a new model...")
            # Compute the base directory (assumes project root is two levels above this file)
            base_dir = os.path.join(os.path.dirname(__file__), '../../')
            data_filepath = os.path.join(base_dir, 'data', 'dataset.csv')
            # Train the model using default hyperparameters (adjust epochs, batch_size, time_steps as needed)
            model, scaler = train_model(data_filepath, model_path, epochs=50, batch_size=32, time_steps=10)
            self.model = model
            self.logger.info("Model training completed and saved.")
        else:
            self.model = load_model(model_path)
        self.threshold_buy = data.get('threshold_buy', 0.6)
        self.threshold_sell = data.get('threshold_sell', 0.4)

    def generate_signal(self, price):
        if not self.model:
            return None
        input_data = np.array([[[price]]])  # shape (1, 1, 1)
        pred = self.model.predict(input_data, verbose=0)[0][0]
        if pred >= self.threshold_buy:
            return 'buy'
        elif pred <= self.threshold_sell:
            return 'sell'
        return None
