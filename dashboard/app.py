import streamlit as st
import pandas as pd
import altair as alt
import requests

st.set_page_config(page_title="Demand Forecast Dashboard", layout="wide")
st.title("📈 Demand Forecast Dashboard")

st.sidebar.header("Prediction Input")
store_id = st.sidebar.text_input("Store ID", "Enter Store ID")
sku_id = st.sidebar.text_input("SKU ID", "Enter SKU ID")
fiscal_month = st.sidebar.text_input("Fiscal Month (yyyymm)", "Enter Month")

st.markdown("---")
st.header("Historical & Predicted Units Sold")

# Load historical data
try:
    df = pd.read_csv("data/train_preprocessed.csv")
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
            # Prepare lists for quantile forecasts
            q10, q50, q90 = [], [], []
            while current_month < target_month:
                current_month = next_month(current_month)
                resp = requests.post(
                    "http://localhost:8000/predict",
                    json={"store_id": store_id, "sku_id": sku_id, "fiscal_month": str(current_month)}
                )
                if resp.status_code == 200:
                    resp_json = resp.json()
                    pred = resp_json.get("prediction")
                    quantile_forecast = resp_json.get("quantile_forecast")
                    future_months.append(str(current_month))
                    future_preds.append(pred)
                    # Extract quantiles for each month from quantile_forecast
                    if quantile_forecast:
                        # quantile_forecast shape: [1, horizon, 10]
                        # Get the last forecasted month (since horizon=1)
                        qf = quantile_forecast[0][len(future_months)-1]
                        q10.append(qf[1])
                        q50.append(qf[5])
                        q90.append(qf[9])
                    else:
                        q10.append(None)
                        q50.append(None)
                        q90.append(None)
                    last_pred = pred
            if future_months:
                pred_df = pd.DataFrame({
                    'fiscal_month': future_months,
                    'units_sold': future_preds,
                    'q10': q10,
                    'q50': q50,
                    'q90': q90
                })
                # Plot median (q50) and quantile band if available
                base = alt.Chart(pred_df)
                pred_line = base.mark_line(point=True, color='orange', strokeDash=[5,5]).encode(
                    x=alt.X('fiscal_month', title='Fiscal Month', sort=None),
                    y=alt.Y('units_sold', title='Units Sold'),
                    tooltip=['fiscal_month', 'units_sold']
                )
                chart = line + pred_line
                # Only plot quantile band and median if quantile data is present and valid
                if all(x is not None for x in q10) and all(x is not None for x in q90):
                    quantile_band = base.mark_area(opacity=0.3, color='orange').encode(
                        x=alt.X('fiscal_month', title='Fiscal Month', sort=None),
                        y=alt.Y('q10', title='10th Percentile'),
                        y2='q90',
                        tooltip=['fiscal_month', 'q10', 'q90']
                    )
                    chart += quantile_band
                if all(x is not None for x in q50):
                    median_line = base.mark_line(color='red').encode(
                        x=alt.X('fiscal_month', title='Fiscal Month', sort=None),
                        y=alt.Y('q50', title='Median (50th)'),
                        tooltip=['fiscal_month', 'q50']
                    )
                    chart += median_line
                st.altair_chart(chart, use_container_width=True)
                st.markdown("""
**Legend:**
- <span style='color:#1f77b4'>●</span> Historical
- <span style='color:orange'>●</span> Point Forecast (dashed orange)
- <span style='color:red'>●</span> Median Forecast (red)
- <span style='color:orange'>■</span> Quantile Band (10th-90th, shaded orange)
""", unsafe_allow_html=True)
    # Only render chart once
        # Display predicted value for the requested fiscal_month
        if fiscal_month and int(fiscal_month) > int(product_df['fiscal_month'].max()):
            try:
                resp = requests.post(
                    "http://localhost:8000/predict",
                    json={"store_id": store_id, "sku_id": sku_id, "fiscal_month": fiscal_month}
                )
                if resp.status_code == 200 and "prediction" in resp.json():
                    pred_value = resp.json()["prediction"]
                    st.success(f"**Predicted units sold for {product_id} in {fiscal_month}: {pred_value:.2f}**")
            except Exception as e:
                st.warning(f"Prediction request failed: {e}")
    else:
        st.info(f"Not enough valid data to plot a line graph for Product ID {product_id}. Please check your input or data file.")
except Exception as e:
    st.warning(f"Could not load historical or prediction data: {e}")

st.markdown("---")
st.caption("© 2025 Demand Forecast Dashboard | Powered by timesfm & FastAPI")
