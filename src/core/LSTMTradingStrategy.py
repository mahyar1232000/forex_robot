"""
LSTMTradingStrategy.py

Implements an advanced LSTM strategy for model training and prediction.
Includes validation monitoring with ReduceLROnPlateau and EarlyStopping based on val_loss.
Trains the LSTM model and saves it as models/lstm_model.h5.
"""

import io
import pandas as pd
import numpy as np
from config import settings
from src.utils.logger import get_logger
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

logger = get_logger(__name__)


class LSTMTradingStrategy:
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path  # e.g., "models/lstm_model.h5"
        self.config = settings.__dict__.copy()
        self.model = None

    def load_data(self):
        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"Loaded data from {self.data_path}, records: {len(df)}")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {self.data_path}: {str(e)}")
            return None

    def preprocess_data(self, df):
        df = df.copy()
        if 'close' not in df.columns:
            logger.error("Data does not contain 'close' column.")
            return None, None
        df['close_norm'] = (df['close'] - df['close'].min()) / (df['close'].max() - df['close'].min())
        sequence_length = self.config.get('lstm_sequence_length', 50)
        X, y = [], []
        for i in range(len(df) - sequence_length):
            X.append(df['close_norm'].iloc[i:i + sequence_length].values)
            y.append(df['close_norm'].iloc[i + sequence_length])
        X = np.array(X)
        y = np.array(y)
        return X, y

    def build_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        model.add(LSTM(32, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse')
        logger.info("LSTM model built successfully with enhanced architecture")
        return model

    def train_model(self, X, y):
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=self.config.get('lstm_patience', 15), restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
        ]
        model = self.build_model((X.shape[1], 1))
        X = X.reshape((X.shape[0], X.shape[1], 1))
        history = model.fit(
            X, y,
            epochs=self.config.get('lstm_epochs', 120),
            batch_size=self.config.get('lstm_batch_size', 128),
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        model.save(self.model_path)
        logger.info(f"LSTM model trained and saved to {self.model_path}")
        final_val_loss = history.history['val_loss'][-1]
        logger.info(f"Final validation loss: {final_val_loss}")
        self.model = model
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + "\n"))
        summary_str = stream.getvalue()
        stream.close()
        return summary_str

    def execute_strategy(self):
        df = self.load_data()
        if df is None or df.empty:
            logger.error("No data available for training")
            return None
        X, y = self.preprocess_data(df)
        if X is None or y is None:
            logger.error("Preprocessing failed.")
            return None
        summary = self.train_model(X, y)
        return summary
