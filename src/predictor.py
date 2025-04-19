import joblib

def load_model_and_predict():
    model = joblib.load("models/random_forest_model.joblib")
    meta = joblib.load("models/feature_metadata.joblib")
    features = meta["features"]
    return model, features
