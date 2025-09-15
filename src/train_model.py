# Model training script

"""
Generic model training module for demand forecasting.
Works with any tabular dataset and target column.
"""

import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import joblib



def train_and_save_model(
    df: pd.DataFrame,
    target_col: str,
    model_path: str = "models/xgboost.joblib",
    feature_cols=None,
    **model_kwargs
) -> tuple:
    """
    Train an XGBoost model and save it. Returns model and validation MAPE.
    """
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in [target_col, 'product_id']]
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(**model_kwargs)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    mape = mean_absolute_percentage_error(y_val, preds)
    print(f"Validation MAPE: {mape}")
    joblib.dump(model, model_path)
    return model, mape

def train_and_validate(train_df: pd.DataFrame, target_col: str = 'units_sold', model_path: str = "models/xgboost.joblib", feature_cols=None, **model_kwargs):
    """
    Train XGBoost model on train_df and validate with a holdout split.
    Returns trained model and validation MAPE.
    """
    return train_and_save_model(train_df, target_col, model_path, feature_cols=feature_cols, **model_kwargs)