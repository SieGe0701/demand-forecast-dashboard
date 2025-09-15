# Prediction script
"""
Generic prediction module for demand forecasting.
Works with any tabular input and trained model.
"""

import joblib
import pandas as pd


def load_model():
    """Load the trained XGBoost model from disk."""
    return joblib.load("models/xgboost.joblib")



def predict(model, input_df: pd.DataFrame):
    """
    Make predictions using the loaded XGBoost model and input DataFrame.
    """
    return model.predict(input_df)

def predict_test_set(model, test_df: pd.DataFrame):
    """Predict target for test set (no target column)."""
    return predict(model, test_df)