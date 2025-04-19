import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Electricity Load Predictor", layout="wide")

# Sidebar UI
st.sidebar.title("⚙️ App Settings")
st.sidebar.markdown("Upload your forecast CSV file to predict electricity load.")

uploaded_file = st.sidebar.file_uploader("Upload Forecast CSV", type=["csv"])
show_raw = st.sidebar.checkbox("Show Raw Uploaded Data")

st.title("⚡ Electricity Load Forecasting Dashboard")
st.markdown("This app uses a machine learning model to predict electricity load based on weather forecasts.")

forecast_days = st.sidebar.selectbox("Select Forecast Range (Days)", [1, 2, 3, 5], index=3)

if uploaded_file:
    input_df = pd.read_csv(uploaded_file, parse_dates=['event_timestamp'])

    if show_raw:
        st.subheader("📄 Raw Uploaded Data")
        st.dataframe(input_df)

    # Load model and features
    model = joblib.load("models/random_forest_model.joblib")
    meta = joblib.load("models/feature_metadata.joblib")
    features = meta["features"]

    # Create missing time features
    input_df["hour"] = input_df["event_timestamp"].dt.hour
    input_df["day_of_week"] = input_df["event_timestamp"].dt.dayofweek
    input_df["is_weekend"] = input_df["day_of_week"] >= 5

    # Filter based on forecast_days
    start_date = input_df['event_timestamp'].min().normalize()
    end_date = start_date + pd.Timedelta(days=forecast_days)
    input_df = input_df[(input_df['event_timestamp'] >= start_date) & (input_df['event_timestamp'] < end_date)]

    # Predict
    X_input = input_df[features]
    predictions = model.predict(X_input)

    results = pd.DataFrame({
        'event_timestamp': input_df['event_timestamp'],
        'predicted_load_MW': predictions
    })

    # Display key stats in columns
    st.subheader("📊 Prediction Summary")
    col1, col2 = st.columns(2)
    col1.metric("Average Load (MW)", f"{results['predicted_load_MW'].mean():.2f}")
    col2.metric("Peak Load (MW)", f"{results['predicted_load_MW'].max():.2f}")

    # Chart output
    st.subheader(f"📈 Load Forecast Over Time: {forecast_days} Days")
    fig = px.line(results, x='event_timestamp', y='predicted_load_MW', title=None)

    # Add vertical lines at each day change
    unique_days = pd.to_datetime(results['event_timestamp']).dt.normalize().drop_duplicates()
    for d in unique_days:
        fig.add_vline(x=d, line_width=1, line_dash="dash", line_color="gray")

    st.plotly_chart(fig, use_container_width=True)

    # Download button
    csv = results.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Forecast as CSV", csv, "predicted_load.csv", "text/csv")

    st.success("✅ Prediction complete and ready to download.")

else:
    st.info("Please upload a CSV file from the sidebar to begin.")

st.sidebar.markdown("---")
st.sidebar.caption("App by Mohamed Sadok Gastli - April 2025")
st.sidebar.markdown("[GitHub](https://github.com/agastli)")
st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/mohamed-sadok-gastli)")
