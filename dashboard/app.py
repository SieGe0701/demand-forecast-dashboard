"""
Streamlit dashboard for demand forecasting.
Works with any tabular dataset.
"""

import os

import streamlit as st

from src.data_preprocessing import clean_data, load_data, transform_data
from src.predict import load_model, predict
from src.train_model import train_and_save_model

st.title("Demand Forecast Dashboard")


# Select train and test files
st.subheader("Select Train and Test Files")
train_file = st.file_uploader("Upload train set (CSV)", type=["csv"], key="train")
test_file = st.file_uploader("Upload test set (CSV)", type=["csv"], key="test")

if train_file:
    train_df = load_data(train_file)
    st.write("Train Data", train_df.head())
    target_col = st.selectbox("Select target column", train_df.columns)
    train_df = clean_data(train_df)
    train_df = transform_data(train_df, target_col=target_col)
    if st.button("Train Model"):
        model, mse = train_and_save_model(train_df, target_col)
        st.success(f"Model trained! Validation MSE: {mse}")

if test_file and os.path.exists("models/model.joblib"):
    test_df = load_data(test_file)
    st.write("Test Data", test_df.head())
    test_df = clean_data(test_df)
    test_df = transform_data(test_df)
    model = load_model()
    preds = predict(model, test_df)
    st.write("Test Set Predictions", preds)
    # Optionally export predictions
    if st.button("Export Predictions"):
        out_df = test_df.copy()
        out_df["predicted_units_sold"] = preds
        out_df.to_csv("data/test_predictions.csv", index=False)
        st.success("Predictions exported to data/test_predictions.csv")
