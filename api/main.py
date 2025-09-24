from fastapi import FastAPI, Request
import pandas as pd
from src.data_preprocessing import preprocess_train_data
from src.train_model import train_and_validate
from src.predict import load_model

app = FastAPI()
model = None

@app.get("/health")
async def health():
    return {"status": "ok"}

# Endpoint to train/retrain model
@app.post("/train")
async def train_endpoint(request: Request):
    payload = await request.json()
    train_path = payload.get("train_path", "data/train.csv")
    target_col = payload.get("target_col", "units_sold")
    global model
    train_df = preprocess_train_data(train_path, target_col=target_col)
    model, mape = train_and_validate(train_df, target_col=target_col)
    return {"message": "Model trained", "validation_mape": mape}

# Endpoint to predict from direct feature input
@app.post("/predict")
async def predict_endpoint(request: Request):
    payload = await request.json()
    store_id = payload.get("store_id")
    sku_id = payload.get("sku_id")
    fiscal_month = payload.get("fiscal_month")  # yyyymm format
    global model
    if model is None:
        model = load_model()
    if store_id is None or sku_id is None or fiscal_month is None:
        return {"error": "store_id, sku_id, and fiscal_month (yyyymm) are required"}
    from src.predict import predict_for_sku_month
    try:
        result = predict_for_sku_month(store_id=store_id, sku_id=sku_id, fiscal_month=fiscal_month)
        print(result)
        return {
            "product_id": result["product_id"],
            "fiscal_month": result["fiscal_month"],
            "prediction": float(result["prediction"]),
            "quantile_forecast": result["quantile_forecast"]
        }
    except Exception as e:
        return {"error": str(e)}
