import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

st.set_page_config(page_title="Electricity Load Predictor", layout="wide")

# Sidebar UI
st.sidebar.title("⚙️ App Settings")
st.sidebar.markdown("Upload your forecast CSV file to predict electricity load.")

uploaded_file = st.sidebar.file_uploader("Upload Forecast CSV", type=["csv"])
show_raw = st.sidebar.checkbox("Show Raw Uploaded Data")

st.title("⚡ Electricity Load Forecasting Dashboard")
st.markdown("This app uses a machine learning model to predict electricity load based on weather forecasts.")

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
    st.subheader("📈 Load Forecast Over Time")
    st.line_chart(results.set_index('event_timestamp'))

    # Download button
    csv = results.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Forecast as CSV", csv, "predicted_load.csv", "text/csv")

    st.success("✅ Prediction complete and ready to download.")

else:
    st.info("Please upload a CSV file from the sidebar to begin.")

st.sidebar.markdown("---")
st.sidebar.caption("App by Mohamed Sadok Gastli- April 2025")
