import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

def test_end_to_end():
    from src.data_preprocessing import preprocess_train_data, preprocess_test_data_with_history, load_data
    from src.train_model import train_and_validate
    from src.predict import load_model, predict_test_set

    print("Starting end-to-end test for demand forecasting pipeline...")
    train_path = "C:/Project/demand-forecast-dashboard/data/train.csv"
    target_col = "units_sold"

    # Check if train data file exists
    if not os.path.exists(train_path):
        print(f"ERROR: {train_path} not found.")
        return

    print("Preprocessing training data...")
    train_df = preprocess_train_data(train_path, target_col=target_col)
    print(f"Train data shape after preprocessing: {train_df.shape}")
    train_df.to_csv("data/train_preprocessed.csv", index=False)

    # Use only training data for validation and predictions
    train_features = [col for col in train_df.columns if col != target_col]

    print("Training and validating model...")
    model, mape = train_and_validate(train_df, target_col=target_col)
    if model is None:
        print("ERROR: Model training failed.")
        return
    print(f"Validation MAPE: {mape}")
    # Print first 5 validation predictions
    if hasattr(model, 'predict'):
        val_preds = model.predict(train_df[train_features])
        print(f"Validation predictions (first 5): {val_preds[:5]}")
        pd.DataFrame(val_preds).to_csv('C:/Project/demand-forecast-dashboard/data/val_preds.csv')

    print("Test completed! Validation predictions saved to data/val_preds.csv.")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    try:
        test_end_to_end()
    except Exception as e:
        import traceback
        print("Exception occurred during test execution:")
        traceback.print_exc()
