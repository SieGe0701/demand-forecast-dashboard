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
    # Create a sample dataset
    df = pd.DataFrame(
        {
            "feature1": np.random.rand(100),
            "feature2": np.random.rand(100),
            "target": np.random.rand(100),
        }
    )
    # Preprocess
    df_clean = clean_data(df)
    df_trans = transform_data(df_clean)
    # Train model
    model, mse = train_and_save_model(
        df_trans, "target", model_path="models/test_model.joblib"
    )
    assert model is not None
    assert mse >= 0
    # Predict
    model_loaded = load_model("models/test_model.joblib")
    input_df = df_trans.drop(columns=["target"])
    preds = predict(model_loaded, input_df)
    assert len(preds) == len(input_df)
    print("End-to-end test passed. Predictions:", preds[:5])


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    test_end_to_end()
