from src.data_preprocessing import preprocess_train_data, preprocess_test_data
from src.train_model import train_and_validate
"""
FastAPI app for serving demand forecasting predictions.
Accepts any tabular input as JSON.
"""

import pandas as pd
from fastapi import FastAPI, Request

from src.predict import load_model, predict


app = FastAPI()
model = None



# Endpoint to train/retrain model
@app.post("/train")
async def train_endpoint(request: Request):
    payload = await request.json()
    train_path = payload.get("train_path", "data/train.csv")
    target_col = payload.get("target_col", "units_sold")
    global model
    train_df = preprocess_train_data(train_path, target_col=target_col)
    model, mse = train_and_validate(train_df, target_col=target_col)
    return {"message": "Model trained", "validation_mse": mse}

# Endpoint to predict on test set
@app.post("/predict_test")
async def predict_test_endpoint(request: Request):
    payload = await request.json()
    test_path = payload.get("test_path", "data/test.csv")
    global model
    if model is None:
        model = load_model()
    test_df = preprocess_test_data(test_path)
    preds = predict(model, test_df)
    return {"predictions": preds.tolist()}
