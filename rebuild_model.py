import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Load the dataset
df = pd.read_csv("weatherkit_plus_load.csv", parse_dates=["event_timestamp"])

# Feature engineering
df["hour"] = df["event_timestamp"].dt.hour
df["day_of_week"] = df["event_timestamp"].dt.dayofweek
df["is_weekend"] = df["day_of_week"] >= 5

# Define features and target
feature_cols = [col for col in df.columns if "forecast" in col] + ["hour", "day_of_week", "is_weekend"]
target_col = "load_MW"

# Train/test split
split_idx = int(len(df) * 0.85)
X_train = df.iloc[:split_idx][feature_cols]
y_train = df.iloc[:split_idx][target_col]

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and metadata
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/random_forest_model.joblib")
joblib.dump({"features": feature_cols, "target": target_col}, "models/feature_metadata.joblib")

print("✅ Model rebuilt and saved successfully in /models")
