"""
Prediction script for demand forecasting.
Takes store_id, sku_id, and fiscal_month as input, creates product_id internally, and returns prediction for product_id and fiscal_month.
"""


import pandas as pd
from src.data_preprocessing import load_data, clean_data

from transformers import pipeline





def load_model(model_name="google/timesfm-1.0"):
    """Load the TimesFM pipeline from Hugging Face."""
    return pipeline("time-series-forecasting", model=model_name)



def predict_for_sku_month(store_id, sku_id, fiscal_month, tsfm_pipe=None, data_path="data/train_preprocessed.csv"):
    """
    Predict demand for a given store_id, sku_id, and fiscal_month using TimesFM pipeline.
    Returns: (product_id, fiscal_month, prediction)
    """
    product_id = f"{store_id}_{sku_id}"
    df = load_data(data_path)
    df = clean_data(df)
    product_df = df[(df["product_id"] == product_id)].copy()
    if product_df.empty:
        raise ValueError(f"No data found for product_id {product_id}")
    # Sort by fiscal_month
    product_df = product_df.sort_values("fiscal_month")
    # Use the last 36 months for context
    context = product_df.tail(36)["units_sold"].tolist()
    # How many months to forecast ahead?
    last_month = int(product_df['fiscal_month'].max())
    target_month = int(fiscal_month)
    horizon = (target_month // 100 - last_month // 100) * 12 + (target_month % 100 - last_month % 100)
    if horizon <= 0:
        raise ValueError("Target month must be after last available month in data.")
    # Load pipeline if not provided
    if tsfm_pipe is None:
        tsfm_pipe = load_model()
    # Run the pipeline
    result = tsfm_pipe(context, prediction_length=horizon)
    forecast = result[0]["prediction"]
    return {
        "product_id": product_id,
        "fiscal_month": target_month,
        "prediction": float(forecast[-1])
    }