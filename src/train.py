import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os


def load_dataset(filepath):
    data = pd.read_csv(filepath)
    return data


def preprocess_data(data, time_steps=10, feature='close'):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[[feature]])
    X, y = [], []
    for i in range(len(data_scaled) - time_steps):
        X.append(data_scaled[i:i + time_steps, 0])
        y.append(data_scaled[i + time_steps, 0])
    X = np.array(X).reshape(-1, time_steps, 1)
    y = np.array(y)
    return X, y, scaler


def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_model(data_filepath, model_save_path, epochs=50, batch_size=32, time_steps=10):
    data = load_dataset(data_filepath)
    X, y, scaler = preprocess_data(data, time_steps=time_steps, feature='close')
    model = build_model((time_steps, 1))
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
    # Ensure the directory exists:
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    return model, scaler


if __name__ == '__main__':
    base_dir = os.path.join(os.path.dirname(__file__), '../')
    data_filepath = os.path.join(base_dir, 'data', 'dataset.csv')
    model_save_path = os.path.join(base_dir, 'models', 'lstm_model.h5')
    train_model(data_filepath, model_save_path)
