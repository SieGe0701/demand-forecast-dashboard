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

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file:
    df = load_data(uploaded_file)
    st.write("Raw Data", df.head())
    df = clean_data(df)
    df = transform_data(df)
    st.write("Processed Data", df.head())
    target_col = st.selectbox("Select target column", df.columns)
    if st.button("Train Model"):
        model, mse = train_and_save_model(df, target_col)
        st.success(f"Model trained! Test MSE: {mse}")
    if os.path.exists("models/model.joblib"):
        model = load_model()
        input_df = df.drop(columns=[target_col])
        preds = predict(model, input_df)
        st.write("Predictions", preds)
