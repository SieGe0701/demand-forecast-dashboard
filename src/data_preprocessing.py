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
    lags: int = 12,
    rolling: int = 7,
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
    from sklearn.preprocessing import StandardScaler

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

    # Encode categorical features
    if encode_categorical:
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            df[col] = df[col].astype('category').cat.codes

    # Handle missing values
    df = df.fillna(df.median(numeric_only=True)).fillna(0)

    # Add interaction features
    if add_interactions:
        num_cols = df.select_dtypes(include=[np.number]).columns
        for i, col1 in enumerate(num_cols):
            for col2 in num_cols[i+1:]:
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]

    # Add custom user-defined features
    if custom_features:
        for name, func in custom_features.items():
            df[name] = func(df)

    # Scale numeric features
    if scale:
        scaler = StandardScaler()
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = scaler.fit_transform(df[num_cols])

    df = df.dropna().reset_index(drop=True)
    return df
