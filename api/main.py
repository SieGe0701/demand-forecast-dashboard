
from fastapi import FastAPI, Request
import pandas as pd
from src.data_preprocessing import preprocess_train_data, preprocess_test_data
from src.train_model import train_and_validate
from src.predict import load_model, predict

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
    sku_id = payload.get("sku_id")
    month = payload.get("month")  # yyyymm format
    global model
    if model is None:
        model = load_model()
    if sku_id is None or month is None:
        return {"error": "sku_id and month (yyyymm) are required"}
    # Parse year and month
    try:
        year = int(str(month)[:4])
        mon = int(str(month)[4:])
    except Exception:
        return {"error": "month must be in yyyymm format"}
    # Load historical data
    import pandas as pd
    from src.data_preprocessing import load_data, clean_data, transform_data
    df = load_data("data/train_preprocessed.csv")
    df = clean_data(df)
    # Filter for the requested sku_id
    sku_df = df[df["sku_id"] == int(sku_id)].copy()
    # Append a new row for the requested month
    new_row = sku_df.iloc[-1:].copy()
    new_row["year"] = year
    new_row["month"] = mon
    sku_df = pd.concat([sku_df, new_row], ignore_index=True)
    # Generate features for the new month
    features_df = transform_data(sku_df, target_col="units_sold", lags=3, rolling=3)
    # Select only the last row (the prediction target)
    input_features = features_df.iloc[[-1]].drop(columns=["units_sold"])
    # Drop 'year' and 'month' if present
    for col in ["year", "month"]:
        if col in input_features.columns:
            input_features = input_features.drop(columns=[col])
    # Encode object columns as category codes
    for col in input_features.select_dtypes(include=['object']).columns:
        input_features[col] = input_features[col].astype('category').cat.codes
    preds = predict(model, input_features)
    return {"predictions": preds.tolist()}
