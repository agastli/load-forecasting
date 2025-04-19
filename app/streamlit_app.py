import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# --- Page Setup ---
st.set_page_config(page_title="Electricity Load Forecasting", layout="wide")

# --- Sidebar ---
st.sidebar.title("⚙️ App Settings")
st.sidebar.markdown("Upload your forecast CSV file to predict electricity load.")

forecast_file = st.sidebar.file_uploader("Upload Forecast CSV", type=["csv"])
show_raw = st.sidebar.checkbox("Show Raw Uploaded Data")

st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Model Controls")
train_file = st.sidebar.file_uploader("Upload Training Data (CSV)", type=["csv"], key="train")
if st.sidebar.button("🔁 Retrain Model"):
    if train_file:
        train_df = pd.read_csv(train_file, parse_dates=['event_timestamp'])
        train_df["hour"] = train_df["event_timestamp"].dt.hour
        train_df["day_of_week"] = train_df["event_timestamp"].dt.dayofweek
        train_df["is_weekend"] = train_df["day_of_week"] >= 5

        feature_cols = [col for col in train_df.columns if col not in ['event_timestamp', 'load_MW']]
        X_train = train_df[feature_cols]
        y_train = train_df['load_MW']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        joblib.dump(model, "models/random_forest_model.joblib")
        joblib.dump({"features": feature_cols}, "models/feature_metadata.joblib")

        st.sidebar.success("✅ Model retrained and saved.")
    else:
        st.sidebar.warning("Please upload training CSV to retrain.")

# Forecast range
forecast_days = st.sidebar.selectbox("Select Forecast Range (Days)", options=[1, 3, 5, 7], index=2)

# --- Main Section ---
st.title("⚡ Electricity Load Forecasting Dashboard")
st.markdown("This app uses a machine learning model to predict electricity load based on weather forecasts.")

if forecast_file:
    input_df = pd.read_csv(forecast_file, parse_dates=['event_timestamp'])

    if show_raw:
        st.subheader("📄 Raw Uploaded Data")
        st.dataframe(input_df)

    model = joblib.load("models/random_forest_model.joblib")
    meta = joblib.load("models/feature_metadata.joblib")
    features = meta["features"]

    # Create derived features
    input_df["hour"] = input_df["event_timestamp"].dt.hour
    input_df["day_of_week"] = input_df["event_timestamp"].dt.dayofweek
    input_df["is_weekend"] = input_df["day_of_week"] >= 5

    # Filter forecast range
    start_time = input_df['event_timestamp'].min()
    end_time = start_time + pd.Timedelta(days=forecast_days)
    input_df = input_df[input_df["event_timestamp"] < end_time]

    # Predict
    X_input = input_df[features]
    predictions = model.predict(X_input)
    results = pd.DataFrame({
        'event_timestamp': input_df['event_timestamp'],
        'predicted_load_MW': predictions
    })

    # --- Display ---
    st.subheader("📊 Prediction Summary")
    col1, col2 = st.columns(2)
    col1.metric("Average Load (MW)", f"{results['predicted_load_MW'].mean():.2f}")
    col2.metric("Peak Load (MW)", f"{results['predicted_load_MW'].max():.2f}")

    st.subheader(f"📈 Load Forecast Over Time: {forecast_days} Days")
    fig = px.line(results, x='event_timestamp', y='predicted_load_MW', labels={
        'event_timestamp': 'Time',
        'predicted_load_MW': 'Predicted Load (MW)'
    })

    # Add vertical lines per day
    unique_dates = results['event_timestamp'].dt.date.unique()
    for date in unique_dates:
        fig.add_vline(x=pd.Timestamp(date), line_dash="dash", line_color="gray")

    st.plotly_chart(fig, use_container_width=True)

    # Download
    csv = results.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Forecast as CSV", csv, "predicted_load.csv", "text/csv")

    st.success("✅ Prediction complete and ready to download.")
else:
    st.info("Please upload a forecast CSV file to begin.")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.caption("App by Mohamed Sadok Gastli • April 2025")
st.sidebar.markdown(
    "🔗 [GitHub](https://github.com/msgastli)  \n"
    "💼 [LinkedIn](https://www.linkedin.com/in/msgastli)  \n"
    "📧 [Email](mailto:msgastli@gmail.com)"
)
