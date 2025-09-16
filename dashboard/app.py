import streamlit as st
import pandas as pd
import altair as alt
import requests

st.set_page_config(page_title="Demand Forecast Dashboard", layout="wide")
st.title("📈 Demand Forecast Dashboard")

st.sidebar.header("Prediction Input")
store_id = st.sidebar.text_input("Store ID", "1")
sku_id = st.sidebar.text_input("SKU ID", "1001")
fiscal_month = st.sidebar.text_input("Fiscal Month (yyyymm)", "202509")

st.markdown("---")
st.header("Historical & Predicted Units Sold")

# Load historical data
try:
    df = pd.read_csv("C:/Project/demand-forecast-dashboard/data/train_preprocessed.csv")
    product_id = f"{store_id}_{sku_id}"
    product_df = df[df["product_id"] == product_id].copy()
    product_df = product_df[pd.to_numeric(product_df['units_sold'], errors='coerce').notnull()]
    product_df['units_sold'] = product_df['units_sold'].astype(float)
    product_df['fiscal_month'] = product_df['fiscal_month'].astype(str)
    # Plot historical data
    if not product_df.empty and len(product_df) > 1:
        line = alt.Chart(product_df).mark_line(point=True).encode(
            x=alt.X('fiscal_month', title='Fiscal Month', sort=None),
            y=alt.Y('units_sold', title='Units Sold'),
            tooltip=['fiscal_month', 'units_sold']
        ).properties(
            title=f"Units Sold: Historical & Predicted for Product {product_id}",
            width=900,
            height=400
        )
        chart = line
        # If prediction requested and future month > last available
        if fiscal_month and int(fiscal_month) > int(product_df['fiscal_month'].max()):
            future_months = []
            future_preds = []
            last_month = int(product_df['fiscal_month'].max())
            target_month = int(fiscal_month)
            def next_month(yyyymm):
                year = yyyymm // 100
                month = yyyymm % 100
                if month == 12:
                    year += 1
                    month = 1
                else:
                    month += 1
                return year * 100 + month
            current_month = last_month
            last_pred = product_df.iloc[-1]['units_sold']
            while current_month < target_month:
                current_month = next_month(current_month)
                resp = requests.post(
                    "http://localhost:8000/predict",
                    json={"store_id": store_id, "sku_id": sku_id, "fiscal_month": str(current_month)}
                )
                if resp.status_code == 200:
                    pred = resp.json()["prediction"]
                    future_months.append(str(current_month))
                    future_preds.append(pred)
                    last_pred = pred
            if future_months:
                pred_df = pd.DataFrame({
                    'fiscal_month': future_months,
                    'units_sold': future_preds
                })
                pred_line = alt.Chart(pred_df).mark_line(point=True, color='orange', strokeDash=[5,5]).encode(
                    x=alt.X('fiscal_month', title='Fiscal Month', sort=None),
                    y=alt.Y('units_sold', title='Units Sold'),
                    tooltip=['fiscal_month', 'units_sold']
                )
                chart = line + pred_line
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info(f"Not enough valid data to plot a line graph for Product ID {product_id}. Please check your input or data file.")
except Exception as e:
    st.warning(f"Could not load historical or prediction data: {e}")

st.markdown("---")
st.caption("© 2025 Demand Forecast Dashboard | Powered by XGBoost & FastAPI")
