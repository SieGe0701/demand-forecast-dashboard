# Data preprocessing script
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

def transform_data(
    df: pd.DataFrame,
    target_col: str = None,
    lags: int = 3,
    rolling: int = 3,
    scale: bool = True,
    encode_categorical: bool = True,
    add_interactions: bool = True,
    custom_features: dict = None
) -> pd.DataFrame:
    """
    Robust feature engineering for demand forecasting:
    - Extract date/time features if a date column exists
    - Add lag and rolling mean features for the target column
    - Encode categorical features
    - Handle missing values
    - Add interaction features
    - Scale numeric features
    - Add custom user-defined features
    """
    import numpy as np

    df = df.copy()

    # Detect date column
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    if date_cols:
        date_col = date_cols[0]
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['dayofweek'] = df[date_col].dt.dayofweek
        df['is_weekend'] = (df[date_col].dt.dayofweek >= 5).astype(int)
        df['quarter'] = df[date_col].dt.quarter
        df['weekofyear'] = df[date_col].dt.isocalendar().week.astype(int)
        # Seasonality
        df['sin_dayofyear'] = np.sin(2 * np.pi * df[date_col].dt.dayofyear / 365)
        df['cos_dayofyear'] = np.cos(2 * np.pi * df[date_col].dt.dayofyear / 365)

    # Add lag and rolling features if target_col is provided
    if target_col and target_col in df.columns:
        for lag in range(1, lags + 1):
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        for win in [rolling, rolling*2]:
            df[f'{target_col}_rolling_mean_{win}'] = df[target_col].rolling(window=win).mean()
            df[f'{target_col}_rolling_std_{win}'] = df[target_col].rolling(window=win).std()
        # Trend feature
        df[f'{target_col}_trend'] = df[target_col] - df[target_col].shift(1)

        # Add ARIMA features
        try:
            from statsmodels.tsa.arima.model import ARIMA
            arima_model = ARIMA(df[target_col], order=(1,0,0)).fit()
            df[f'{target_col}_arima_fitted'] = arima_model.fittedvalues
            df[f'{target_col}_arima_resid'] = arima_model.resid
        except Exception as e:
            print(f"ARIMA feature generation failed: {e}")

    # Encode categorical features
    if encode_categorical:
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            df[col] = df[col].astype('category').cat.codes

    # Handle missing values
    df = df.fillna(df.median(numeric_only=True)).fillna(0)

    # Exclude all interaction features for now

    # Add custom user-defined features
    if custom_features:
        for name, func in custom_features.items():
            df[name] = func(df)

    # Feature scaling removed

    df = df.dropna().reset_index(drop=True)
    return df

def preprocess_train_data(filepath: str, target_col: str = 'units_sold') -> pd.DataFrame:
    """Load, clean, and transform training data."""
    df = load_data(filepath)
    df = clean_data(df)
    df = transform_data(df, target_col=target_col)
    return df

def preprocess_test_data(filepath: str) -> pd.DataFrame:
    """Load, clean, and transform test data (no target)."""
    df = load_data(filepath)
    df = clean_data(df)
    df = transform_data(df)
    return df

def preprocess_test_data_with_history(test_filepath: str, train_df: pd.DataFrame, target_col: str = 'units_sold', lags: int = 12, rolling: int = 7) -> pd.DataFrame:
    """
    Preprocess test data using train history for lag/rolling features.
    Appends last lags rows of train_df to test_df, computes features, then returns only test rows.
    """
    test_df = load_data(test_filepath)
    test_df = clean_data(test_df)
    # Get last lags rows from train_df
    history_df = train_df.tail(lags).copy()
    # Concatenate history and test
    combined_df = pd.concat([history_df, test_df], ignore_index=True)
    # Compute features
    combined_df = transform_data(combined_df, target_col=target_col, lags=lags, rolling=rolling)
    # Remove history rows
    test_features_df = combined_df.iloc[lags:].reset_index(drop=True)
    # Drop target column if present
    if target_col in test_features_df.columns:
        test_features_df = test_features_df.drop(columns=[target_col])
    return test_features_df