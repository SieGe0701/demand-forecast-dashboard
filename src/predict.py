# Prediction script
"""
Generic prediction module for demand forecasting.
Works with any tabular input and trained model.
"""

import joblib
import pandas as pd


def load_model(model_path: str = "models/model.joblib"):
    """Load a trained model from disk."""
    return joblib.load(model_path)


def predict(model, input_df: pd.DataFrame):
    """Make predictions using the loaded model and input DataFrame."""
    return model.predict(input_df)
