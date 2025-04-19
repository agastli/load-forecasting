import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

st.set_page_config(page_title="Electricity Load Predictor", layout="centered")

st.title("Electricity Load Forecasting App")
st.write("Upload forecasted weather data to predict next day's load.")

uploaded_file = st.file_uploader("Upload your weather forecast CSV file", type=["csv"])

if uploaded_file:
    input_df = pd.read_csv(uploaded_file, parse_dates=['event_timestamp'])
    st.write("Preview of uploaded data:")
    st.dataframe(input_df.head())

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

    # Summary Stats
    st.subheader("Prediction Summary")
    st.metric("Average Load (MW)", f"{results['predicted_load_MW'].mean():.2f}")
    st.metric("Peak Load (MW)", f"{results['predicted_load_MW'].max():.2f}")

    # Line Chart
    st.subheader("Predicted Load Over Time")
    st.line_chart(results.set_index('event_timestamp'))

    # Download Button
    csv = results.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions as CSV", csv, "predicted_load.csv", "text/csv")

    st.success("Prediction complete and ready to download!")

st.sidebar.markdown("---")
st.sidebar.write("Built with by CodeGPT")
