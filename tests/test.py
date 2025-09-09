"""
End-to-end test for demand forecasting pipeline.
Loads sample data, preprocesses, trains, predicts, and checks output shape.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from src.data_preprocessing import clean_data, transform_data
from src.predict import load_model, predict
from src.train_model import train_and_save_model



def test_end_to_end():
    from src.data_preprocessing import preprocess_train_data, preprocess_test_data
    from src.train_model import train_and_validate
    from src.predict import load_model, predict_test_set

    # Paths
    train_path = "data/train.csv"
    test_path = "data/test.csv"
    model_path = "models/test_model.joblib"
    target_col = "units_sold"

    # Preprocess train and test
    train_df = preprocess_train_data(train_path, target_col=target_col)
    test_df = preprocess_test_data(test_path)

    # Train and validate
    model, mse = train_and_validate(train_df, target_col=target_col, model_path=model_path)
    assert model is not None
    assert mse >= 0
    print(f"Validation MSE: {mse}")

    # Predict on test set
    model_loaded = load_model(model_path)
    preds = predict_test_set(model_loaded, test_df)
    assert len(preds) == len(test_df)
    print("Test set predictions (first 5):", preds[:5])


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    test_end_to_end()
