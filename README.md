# Demand Forecast Dashboard (End-to-End Solution)

## Overview
A robust, production-ready demand forecasting solution with:
- Data preprocessing and feature engineering
- XGBoost model training and validation
- FastAPI for real-time predictions
- Streamlit dashboard for visualization
- Sequential forecasting for future months

---

## 1. Setup

### Prerequisites
- Python 3.8+
- (Recommended) Create a virtual environment:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\activate
  ```
- Install dependencies:
  ```powershell
  pip install -r requirements.txt
  ```

---

## 2. Data Preprocessing

- Place your raw data in `data/train.csv`.
- Run preprocessing (optional, handled in test script):
  ```powershell
  python -c "from src.data_preprocessing import preprocess_train_data; df = preprocess_train_data('data/train.csv', target_col='units_sold'); df.to_csv('data/train_preprocessed.csv', index=False)"
  ```

---

## 3. Model Training & Validation

- Run the end-to-end test (preprocess, train, validate, predict):
  ```powershell
  python tests/test.py
  ```
- Model is saved to `models/xgboost.joblib`.
- Validation predictions are saved to `data/val_preds.csv`.

---

## 4. Start the API

- Launch FastAPI server:
  ```powershell
  .venv\Scripts\uvicorn api.main:app --reload
  ```
- Health check: [http://localhost:8000/health](http://localhost:8000/health)
- Predict endpoint: `POST /predict` with JSON body:
  ```json
  { "store_id": "1", "sku_id": "1001", "fiscal_month": "202509" }
  ```

---

## 5. Start the Dashboard

- In a new terminal:
  ```powershell
  streamlit run dashboard/app.py
  ```
- Enter `store_id`, `sku_id`, and `fiscal_month` in the sidebar.
- View historical and predicted demand in a single line graph.

---

## 6. Sequential Forecasting
- If you request a future month (e.g., 202509), the model will sequentially predict for each missing month, using each prediction as input for the next, until the target month is reached.

---

## 7. Project Structure
```
├── api/
│   └── main.py           # FastAPI app
├── dashboard/
│   └── app.py            # Streamlit dashboard
├── data/
│   ├── train.csv         # Raw data
│   ├── train_preprocessed.csv # Preprocessed data
│   └── val_preds.csv     # Validation predictions
├── models/
│   └── xgboost.joblib    # Trained model
├── src/
│   ├── data_preprocessing.py
│   ├── predict.py
│   └── train_model.py
├── tests/
│   └── test.py           # End-to-end test
├── requirements.txt
├── Dockerfile (optional)
└── README.md
```

---

## 8. Tips
- Make sure the API server is running before using the dashboard.
- Use the test script to validate the full pipeline after any code/data changes.
- For production, consider Dockerizing and adding CI/CD.

---

## 9. License
MIT

