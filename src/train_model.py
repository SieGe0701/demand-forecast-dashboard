# Model training script

"""
Generic model training module for demand forecasting.
Works with any tabular dataset and target column.
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def train_and_save_model(
    df: pd.DataFrame, target_col: str, model_path: str = "models/model.joblib"
):
    """Train a RandomForestRegressor and save the model."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"Test MSE: {mse}")
    joblib.dump(model, model_path)
    return model, mse
