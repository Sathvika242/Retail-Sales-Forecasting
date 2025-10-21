import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
from datetime import timedelta

warnings.filterwarnings("ignore")

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(page_title="Retail Sales Forecast Dashboard", layout="wide")
st.title("🎯 Retail Sales Forecast Dashboard")

# --------------------------------
# LOAD DATA
# --------------------------------
CSV_PATH = r"Walmart.csv"  

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    if 'IsHoliday' in df.columns:
        df['IsHoliday'] = df['IsHoliday'].astype(int)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    return df

df = load_data()
st.dataframe(df.head())

# --------------------------------
# SIDEBAR FILTERS
# --------------------------------
st.sidebar.header("🧭 Filters")

store_options = df['Store'].unique()
selected_stores = st.sidebar.multiselect("🏪 Select Store(s)", store_options, default=store_options[:1])
forecast_weeks = st.sidebar.slider("📅 Forecast Horizon (Weeks)", 4, 20, 12, 1)

# Optional filters
fuel_range = (df['Fuel_Price'].min(), df['Fuel_Price'].max()) if 'Fuel_Price' in df.columns else (0, 1000)
unemp_range = (df['Unemployment'].min(), df['Unemployment'].max()) if 'Unemployment' in df.columns else (0, 100)

if 'Fuel_Price' in df.columns:
    fuel_range = st.sidebar.slider(
        "⛽ Fuel Price Range",
        float(df['Fuel_Price'].min()), float(df['Fuel_Price'].max()), fuel_range
    )

if 'Unemployment' in df.columns:
    unemp_range = st.sidebar.slider(
        "💼 Unemployment Rate Range",
        float(df['Unemployment'].min()), float(df['Unemployment'].max()), unemp_range
    )

filtered_data = df[df['Store'].isin(selected_stores)]

if 'Fuel_Price' in filtered_data.columns:
    filtered_data = filtered_data[
        (filtered_data['Fuel_Price'] >= fuel_range[0]) &
        (filtered_data['Fuel_Price'] <= fuel_range[1])
    ]

if 'Unemployment' in filtered_data.columns:
    filtered_data = filtered_data[
        (filtered_data['Unemployment'] >= unemp_range[0]) &
        (filtered_data['Unemployment'] <= unemp_range[1])
    ]

# --------------------------------
# METRICS
# --------------------------------
st.markdown("## 📊 Key Metrics")
col1, col2, col3 = st.columns(3)

total_sales = filtered_data['Weekly_Sales'].sum()
avg_sales = filtered_data['Weekly_Sales'].mean()
store_count = len(selected_stores)

col1.metric("💰 Total Sales", f"₹{total_sales:,.2f}")
col2.metric("📈 Average Weekly Sales", f"₹{avg_sales:,.2f}")
col3.metric("🏪 Stores Selected", f"{store_count}")

st.markdown("---")

# --------------------------------
# SALES TREND
# --------------------------------
st.subheader("📅 Weekly Sales Trend")
trend_fig = px.line(filtered_data, x='Date', y='Weekly_Sales', color='Store', markers=True,
                    title="Weekly Sales Trend by Store")
st.plotly_chart(trend_fig, use_container_width=True)

# --------------------------------
# PREDICT BUTTON
# --------------------------------
predict = st.sidebar.button("🔮 Predict")

if predict:
    st.subheader(f"🔮 {forecast_weeks}-Week Sales Forecast (SARIMA)")
    forecast_dfs = []
    comparison_dfs = []

    for store in selected_stores:
        store_data = filtered_data[filtered_data['Store'] == store].sort_values('Date')
        sales_series = store_data.set_index('Date')['Weekly_Sales']

        # Exogenous variables (if present)
        exog_cols = [c for c in ['Fuel_Price', 'Unemployment', 'IsHoliday'] if c in store_data.columns]
        exog = store_data[exog_cols].copy() if exog_cols else None
        if exog is not None:
            exog.index = store_data['Date']

        if len(sales_series) < 10:
            st.warning(f"⚠️ Not enough data for Store {store}")
            continue

        # Fit SARIMA
        model = SARIMAX(sales_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52), exog=exog)
        model_fit = model.fit(disp=False)

        # Future exogenous data
        if exog is not None:
            last_exog = exog.iloc[[-1]].copy()
            future_exog = pd.concat([last_exog] * forecast_weeks, ignore_index=True)
            future_exog.index = pd.date_range(
                start=sales_series.index.max() + pd.Timedelta(weeks=1),
                periods=forecast_weeks, freq='W'
            )
        else:
            future_exog = None

        # Forecast future sales
        future_dates = pd.date_range(
            start=sales_series.index.max() + pd.Timedelta(weeks=1),
            periods=forecast_weeks, freq='W'
        )
        forecast = model_fit.forecast(steps=forecast_weeks, exog=future_exog)

        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Forecasted_Sales': forecast.values,
            'Store': store
        })
        forecast_dfs.append(forecast_df)

        # Combine actual and forecast for plotting
        actual_df = store_data[['Date', 'Weekly_Sales']].copy()
        actual_df.rename(columns={'Weekly_Sales': 'Sales'}, inplace=True)
        actual_df['Type'] = 'Actual'
        actual_df['Store'] = store

        forecast_plot_df = forecast_df.rename(columns={'Forecasted_Sales': 'Sales'})
        forecast_plot_df['Type'] = 'Forecast'

        comparison_dfs.append(pd.concat([actual_df, forecast_plot_df], ignore_index=True))

    forecast_all = pd.concat(forecast_dfs)
    comparison_all = pd.concat(comparison_dfs)

    # --------------------------------
    # GRAPH 1: Actual vs Forecast
    # --------------------------------
    st.subheader("📊 Actual vs Forecasted Sales")
    fig1 = px.line(comparison_all, x='Date', y='Sales', color='Store',
                   line_dash='Type', markers=True, title="Actual vs Forecasted Sales by Store")
    st.plotly_chart(fig1, use_container_width=True)

    # --------------------------------
    # GRAPH 2: Forecast-Only with Trendline
    # --------------------------------
    st.subheader("📈 Forecasted Sales with Trendline (Future Weeks Only)")
    fig2 = px.scatter(
        forecast_all,
        x='Date',
        y='Forecasted_Sales',
        color='Store',
        trendline='ols',  # Adds linear trendline
        title="Forecasted Sales Trend with Linear Trendline"
    )
    fig2.update_traces(mode='lines+markers')
    st.plotly_chart(fig2, use_container_width=True)

    # --------------------------------
    # FORECAST TABLE + DOWNLOAD
    # --------------------------------
    st.subheader("📋 Forecast Table")
    forecast_table = forecast_all.copy()
    forecast_table['Forecasted_Sales'] = forecast_table['Forecasted_Sales'].apply(lambda x: f"₹{x:,.2f}")
    forecast_table.insert(0, "S.No", range(1, len(forecast_table) + 1))
    st.dataframe(forecast_table)

    csv = forecast_all.to_csv(index=False)
    st.download_button("📥 Download Forecast CSV", data=csv, file_name="forecast_sarima.csv", mime="text/csv")
