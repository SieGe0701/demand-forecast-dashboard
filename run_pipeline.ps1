# PowerShell script for Demand Forecast Dashboard end-to-end workflow

param(
    [switch]$Preprocess,
    [switch]$Train,
    [switch]$API,
    [switch]$Dashboard,
    [switch]$Test,
    [switch]$Clean
)

if ($Preprocess) {
    Write-Host "[Preprocessing] Running data preprocessing..."
    .\.venv\Scripts\python -c "from src.data_preprocessing import preprocess_train_data; df = preprocess_train_data('data/train.csv', target_col='units_sold'); df.to_csv('data/train_preprocessed.csv', index=False)"
}

if ($Train) {
    Write-Host "[Training] Running model training and validation..."
    .\.venv\Scripts\python tests/test.py
}

if ($API) {
    Write-Host "[API] Starting FastAPI server..."
    .\.venv\Scripts\uvicorn api.main:app --reload
}

if ($Dashboard) {
    Write-Host "[Dashboard] Starting Streamlit dashboard..."
    streamlit run dashboard/app.py
}

if ($Test) {
    Write-Host "[Test] Running end-to-end test..."
    .\.venv\Scripts\python tests/test.py
}

if ($Clean) {
    Write-Host "[Clean] Removing generated files..."
    Remove-Item -ErrorAction SilentlyContinue data\train_preprocessed.csv, data\val_preds.csv
    Remove-Item -ErrorAction SilentlyContinue models\xgboost.joblib
}

if (-not ($Preprocess -or $Train -or $API -or $Dashboard -or $Test -or $Clean)) {
    Write-Host "Usage:"
    Write-Host ".\run_pipeline.ps1 -Preprocess -Train -API -Dashboard -Test -Clean"
    Write-Host "Use any combination of switches to run specific steps."
}
