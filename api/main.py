"""
FastAPI app for serving demand forecasting predictions.
Accepts any tabular input as JSON.
"""

import pandas as pd
from fastapi import FastAPI, Request

from src.predict import load_model, predict

app = FastAPI()
model = load_model()


@app.post("/predict")
async def predict_endpoint(request: Request):
    data = await request.json()
    input_df = pd.DataFrame(data)
    preds = predict(model, input_df)
    return {"predictions": preds.tolist()}
