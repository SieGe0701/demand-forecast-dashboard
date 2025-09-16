"""
Prediction script for demand forecasting.
Takes store_id, sku_id, and fiscal_month as input, creates product_id internally, and returns prediction for product_id and fiscal_month.
"""

import pandas as pd
import joblib
from src.data_preprocessing import load_data, clean_data, transform_data



def load_model(model_path="models/xgboost.joblib"):
    """Load the trained XGBoost model from disk."""
    return joblib.load(model_path)

def predict_for_sku_month(store_id, sku_id, fiscal_month, model=None, data_path="data/train_preprocessed.csv"):
    """
    Predict demand for a given store_id, sku_id, and fiscal_month.
    If fiscal_month is in the future, sequentially predict for each missing month,
    appending each prediction to the data and generating features for the next step.
    Returns: (product_id, fiscal_month, prediction)
    """
    # Create product_id
    product_id = f"{store_id}_{sku_id}"
    # Load preprocessed data
    df = load_data(data_path)
    df = clean_data(df)
    # Filter for product_id
    product_df = df[(df["product_id"] == product_id)].copy()
    if product_df.empty:
        raise ValueError(f"No data found for product_id {product_id}")
    # Ensure store_id and sku_id columns exist after filtering
    if 'store_id' not in product_df.columns or 'sku_id' not in product_df.columns:
        product_df[['store_id', 'sku_id']] = product_df['product_id'].str.split('_', expand=True)
        product_df['store_id'] = product_df['store_id'].astype(int)
        product_df['sku_id'] = product_df['sku_id'].astype(int)
    # Find last available fiscal_month
    last_month = int(product_df['fiscal_month'].max())
    target_month = int(fiscal_month)
    # Helper to increment fiscal_month
    def next_month(yyyymm):
        year = yyyymm // 100
        month = yyyymm % 100
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
        return year * 100 + month
    # Sequentially predict for each missing month
    current_month = last_month
    while current_month < target_month:
        new_row = product_df.iloc[-1:].copy()
        current_month = next_month(current_month)
        new_row["fiscal_month"] = current_month
        product_df = pd.concat([product_df, new_row], ignore_index=True)
        # Feature engineering for prediction
        features_df = transform_data(product_df, target_col="units_sold", lags=3, rolling=3)
        # Select only the last row (the prediction target)
        input_features = features_df.iloc[[-1]].drop(columns=["units_sold", "product_id"], errors="ignore")
        # Drop 'year' and 'month' if present
        for col in ["year", "month"]:
            if col in input_features.columns:
                input_features = input_features.drop(columns=[col])
        # Encode object columns as category codes
        for col in input_features.select_dtypes(include=['object']).columns:
            input_features[col] = input_features[col].astype('category').cat.codes
        # Load model if not provided
        if model is None:
            model = load_model()
        # Predict
        pred = model.predict(input_features)[0]
        # Set predicted value for next step
        product_df.at[product_df.index[-1], 'units_sold'] = pred
    # Final prediction for target_month
    return {
        "product_id": product_id,
        "fiscal_month": target_month,
        "prediction": float(product_df.iloc[-1]['units_sold'])
    }