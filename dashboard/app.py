
import streamlit as st
import pandas as pd
import requests
import plotly.express as px

st.set_page_config(page_title="Demand Forecast Dashboard", layout="wide")
st.title("📈 Demand Forecast Dashboard")


st.sidebar.header("Prediction Input")
sku_id = st.sidebar.text_input("SKU ID", "1001")
month = st.sidebar.text_input("Month (yyyymm)", "202509")

if st.sidebar.button("Predict Demand"):
    with st.spinner("Requesting prediction from API..."):
        try:
            response = requests.post(
                "http://localhost:8000/predict",
                json={"sku_id": sku_id, "month": month}
            )
            if response.status_code == 200:
                preds = response.json()["predictions"]
                st.success(f"Predicted units sold: {preds[0]}")
            else:
                st.error(f"API error: {response.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")

st.markdown("---")
st.header("Historical Performance & Trends")

# Load and plot historical data (customize path and columns)
try:
    df = pd.read_csv("C:/Project/demand-forecast-dashboard/data/train_preprocessed.csv")
    if sku_id:
        sku_df = df[df["sku_id"] == int(sku_id)]
        if not sku_df.empty:
            fig = px.line(sku_df, x="week", y="units_sold", title=f"Units Sold Over Time for SKU {sku_id}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No historical data for SKU {sku_id}.")
    else:
        fig = px.line(df, x="week", y="units_sold", title="Units Sold Over Time (by Week)")
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Feature Correlations")
    corr = df.corr()
    st.dataframe(corr)
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Feature Correlation Heatmap")
    st.plotly_chart(fig_corr, use_container_width=True)
except Exception as e:
    st.warning(f"Could not load historical data: {e}")

st.markdown("---")
st.header("Model Validation")
try:
    val_preds = pd.read_csv("C:/Project/demand-forecast-dashboard/data/val_preds.csv")
    st.subheader("Sample Validation Predictions")
    st.dataframe(val_preds.head(10))
except Exception as e:
    st.info("No validation predictions found.")

st.markdown("---")
st.caption("© 2025 Demand Forecast Dashboard | Powered by XGBoost & FastAPI")
