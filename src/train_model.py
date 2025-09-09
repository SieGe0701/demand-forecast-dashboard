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
    df: pd.DataFrame,
    target_col: str,
    model_path: str = "models/model.joblib",
    model_type: str = "random_forest",
    **model_kwargs
) -> tuple:
    """
    Train a demand forecasting model and save it.
    Supports RandomForestRegressor (default), with extensibility for other models.
    Prints feature importances if available.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import GradientBoostingRegressor

    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if model_type == "random_forest":
        model = RandomForestRegressor(**model_kwargs)
    elif model_type == "linear":
        model = LinearRegression(**model_kwargs)
    elif model_type == "gbr":
        model = GradientBoostingRegressor(**model_kwargs)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"Test MSE: {mse}")
    if hasattr(model, "feature_importances_"):
        print("Feature importances:")
        for name, imp in zip(X.columns, model.feature_importances_):
            print(f"  {name}: {imp:.4f}")
    joblib.dump(model, model_path)
    return model, mse
# ...existing code...

def train_and_validate(train_df: pd.DataFrame, target_col: str = 'units_sold', model_path: str = 'models/model.joblib', model_type: str = 'random_forest', **model_kwargs):
    """
    Train model on train_df and validate with a holdout split.
    Returns trained model and validation MSE.
    """
    return train_and_save_model(train_df, target_col, model_path, model_type, **model_kwargs)