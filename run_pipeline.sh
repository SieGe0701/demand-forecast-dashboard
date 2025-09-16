#!/bin/bash
# Bash script for Demand Forecast Dashboard end-to-end workflow

show_help() {
  echo "Usage: $0 [preprocess] [train] [api] [dashboard] [test] [clean]"
  echo "Run any combination of steps by listing them as arguments."
}

for arg in "$@"
do
  case $arg in
    preprocess)
      echo "[Preprocessing] Running data preprocessing..."
      ./.venv/bin/python -c "from src.data_preprocessing import preprocess_train_data; df = preprocess_train_data('data/train.csv', target_col='units_sold'); df.to_csv('data/train_preprocessed.csv', index=False)"
      ;;
    train)
      echo "[Training] Running model training and validation..."
      ./.venv/bin/python tests/test.py
      ;;
    api)
      echo "[API] Starting FastAPI server..."
      ./.venv/bin/uvicorn api.main:app --reload
      ;;
    dashboard)
      echo "[Dashboard] Starting Streamlit dashboard..."
      streamlit run dashboard/app.py
      ;;
    test)
      echo "[Test] Running end-to-end test..."
      ./.venv/bin/python tests/test.py
      ;;
    clean)
      echo "[Clean] Removing generated files..."
      rm -f data/train_preprocessed.csv data/val_preds.csv
      rm -f models/xgboost.joblib
      ;;
    *)
      show_help
      exit 1
      ;;
  esac
done

if [ $# -eq 0 ]; then
  show_help
fi
