"""
model_manager.py
================
Provides functions for saving, loading, and managing the trained models.
"""

from tensorflow.keras.models import load_model


def save_model(model, filepath):
    model.save(filepath)


def load_trained_model(filepath):
    return load_model(filepath)
