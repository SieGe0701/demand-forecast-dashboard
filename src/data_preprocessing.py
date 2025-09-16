"""
Data preprocessing module for demand forecasting.
- Converts week/date to fiscal_month
- Aggregates raw data by store_id, sku_id, fiscal_month
- Applies robust feature engineering
"""
import pandas as pd
import numpy as np

def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from a file path."""
    return pd.read_csv(filepath)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop NA, reset index."""
    return df.dropna().reset_index(drop=True)

def transform_data(
    df: pd.DataFrame,
    target_col: str = None,
    lags: int = 12,
    rolling: int = 3,
    encode_categorical: bool = True,
    custom_features: dict = None
) -> pd.DataFrame:
    """
    Preprocessing pipeline:
    1. Convert week/date to fiscal_month
    2. Aggregate by store_id, sku_id, fiscal_month
    3. Feature engineering (lag, rolling, trend, ARIMA)
    4. Encode categoricals, handle missing, add custom features
    """
    df = df.copy()

    # Step 1: Convert week/date to fiscal_month
    if 'week' in df.columns:
        df['week'] = pd.to_datetime(df['week'])
        df['year'] = df['week'].dt.year
        df['month'] = df['week'].dt.month
        df['fiscal_month'] = df['year'] * 100 + df['month']

    # Step 1.5: Create a Product id based on store and sku_id
    df['product_id'] = df['store_id'].astype(str) + '_' + df['sku_id'].astype(str)

    # Step 2: Aggregate raw data by product_id, fiscal_month
    if all(col in df.columns for col in ['product_id', 'fiscal_month', target_col]):
        agg_cols = ['product_id', 'fiscal_month']
        df = df.groupby(agg_cols, as_index=False).agg({target_col: 'sum'})
        # After aggregation, split product_id back into store_id and sku_id
        df[['store_id', 'sku_id']] = df['product_id'].str.split('_', expand=True)
        df['store_id'] = df['store_id'].astype(int)
        df['sku_id'] = df['sku_id'].astype(int)

    # Step 3: Feature engineering
    if target_col and target_col in df.columns:
        for lag in range(1, lags + 1):
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        for win in [rolling, rolling*2]:
            df[f'{target_col}_rolling_mean_{win}'] = df[target_col].rolling(window=win).mean()
            df[f'{target_col}_rolling_std_{win}'] = df[target_col].rolling(window=win).std()
        df[f'{target_col}_trend'] = df[target_col] - df[target_col].shift(1)
        # ARIMA features
        try:
            from statsmodels.tsa.arima.model import ARIMA
            arima_model = ARIMA(df[target_col], order=(1,0,0)).fit()
            df[f'{target_col}_arima_fitted'] = arima_model.fittedvalues
            df[f'{target_col}_arima_resid'] = arima_model.resid
        except Exception as e:
            print(f"ARIMA feature generation failed: {e}")

    # Step 4: Encode categoricals
    if encode_categorical:
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        # Exclude product_id from encoding
        cat_cols = [col for col in cat_cols if col != 'product_id']
        for col in cat_cols:
            df[col] = df[col].astype('category').cat.codes

    # Step 5: Handle missing values
    df = df.fillna(df.median(numeric_only=True)).fillna(0)

    # Step 6: Add custom user-defined features
    if custom_features:
        for name, func in custom_features.items():
            df[name] = func(df)

    # Final cleanup
    df = df.dropna().reset_index(drop=True)
    # Remove product_id from training features if present
    if 'product_id' in df.columns:
        train_df = df.drop(columns=['product_id'])
        # But keep product_id in the returned DataFrame for prediction
        df = pd.concat([train_df, df['product_id']], axis=1)
    df = df.drop(columns=['week', 'weekofyear'], errors='ignore')
    return df

def preprocess_train_data(
    filepath: str,
    target_col: str = 'units_sold',
    lags: int = 3,
    rolling: int = 3,
    encode_categorical: bool = True,
    custom_features: dict = None
) -> pd.DataFrame:
    """
    Loads raw training data, applies preprocessing pipeline, and returns processed DataFrame.
    """
    df = load_data(filepath)
    df = clean_data(df=df)
    df = transform_data(
        df,
        target_col=target_col,
        lags=lags,
        rolling=rolling,
        encode_categorical=encode_categorical,
        custom_features=custom_features
    )
    return df
