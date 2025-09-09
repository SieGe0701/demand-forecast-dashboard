# Data preprocessing script
# placeholder

"""
Generic data preprocessing module for demand forecasting.
Functions can be used with any tabular dataset (CSV, Excel, etc).
"""

import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from a file path."""
    return pd.read_csv(filepath)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop NA, reset index."""
    df = df.dropna().reset_index(drop=True)
    return df


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Placeholder for feature engineering and transformations."""
    # Add your transformations here
    return df
