"""
Prediction script for demand forecasting.
Takes store_id, sku_id, and fiscal_month as input, creates product_id internally, and returns prediction for product_id and fiscal_month.
"""


import pandas as pd
from src.data_preprocessing import load_data, clean_data
from transformers import pipeline
import timesfm


def load_model(model_name="google/timesfm-1.0"):
    """Load the TimesFM pipeline from Hugging Face."""
    model = timesfm.TimesFM_2p5_200M_torch()
    model.load_checkpoint()
    model.compile(
        timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=256,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
    )
    return model



def predict_for_sku_month(store_id, sku_id, fiscal_month, data_path="data/train_preprocessed.csv"):
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
    model = load_model()
    # Run the pipeline
    point_forecast, quantile_forecast = model.forecast(horizon=horizon, inputs=[context])
    prediction = float(point_forecast[0, -1])
    return {
        "product_id": product_id,
        "fiscal_month": target_month,
        "prediction": prediction,
        "quantile_forecast": quantile_forecast.tolist()
    }