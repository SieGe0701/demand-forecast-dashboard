
import requests

API_URL = "http://localhost:8000/predict"
DASHBOARD_URL = "http://localhost:8501"

def test_api_predict():
    payload = {"store_id": "8023", "sku_id": "216233", "fiscal_month": "201404"}
    resp = requests.post(API_URL, json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    assert "quantile_forecast" in data
    assert isinstance(data["prediction"], (int, float))
    assert isinstance(data["quantile_forecast"], list)
    print("API /predict endpoint test passed.")

def test_api_invalid_input():
    payload = {"store_id": "", "sku_id": "", "fiscal_month": ""}
    resp = requests.post(API_URL, json=payload)
    assert resp.status_code in [400, 422]
    print("API invalid input test passed.")

def test_dashboard_running():
    resp = requests.get(DASHBOARD_URL)
    assert resp.status_code == 200
    assert "Demand Forecast Dashboard" in resp.text
    print("Dashboard running test passed.")

if __name__ == "__main__":
    test_api_predict()
    test_api_invalid_input()
    test_dashboard_running()
