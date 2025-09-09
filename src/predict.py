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



def predict(model, input_df: pd.DataFrame, return_interval: bool = False, alpha: float = 0.05):
    """
    Make predictions using the loaded model and input DataFrame.
    Optionally return prediction intervals for supported models.
    """
    preds = model.predict(input_df)
    if return_interval:
        # For RandomForestRegressor, use percentiles of trees for intervals
        if hasattr(model, 'estimators_'):
            all_preds = [tree.predict(input_df) for tree in model.estimators_]
            lower = pd.DataFrame(all_preds).quantile(alpha/2, axis=0)
            upper = pd.DataFrame(all_preds).quantile(1-alpha/2, axis=0)
            return preds, lower.values, upper.values
        else:
            # For other models, return None for intervals
            return preds, None, None
    return preds

def predict_test_set(model, test_df: pd.DataFrame):
    """Predict target for test set (no target column)."""
    return predict(model, test_df)