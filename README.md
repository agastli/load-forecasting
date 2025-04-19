# ⚡ Electricity Load Forecasting App

Forecast next-day electricity demand using forecasted weather data.  
Built with **Python, scikit-learn, Streamlit**, and deployed on **Streamlit Cloud**.

## 📌 Features

- ✅ Upload your own forecasted weather CSV
- 📊 Automatically extracts temporal features (hour, day of week, weekend)
- 🤖 Predicts load using a pre-trained **Random Forest** model
- 📈 Visualizes results in an interactive line chart
- 📥 Download predicted load as a CSV file

## 🧠 Model

- Trained using `RandomForestRegressor` from `scikit-learn`
- Input features include:
  - Weather observations (temperature, humidity, etc.)
  - Time-based features: `hour`, `day_of_week`, `is_weekend`
- Stored in: `models/random_forest_model.joblib`

## 🚀 Deployment

- Code hosted on GitHub
- App deployed on [Streamlit Cloud](https://streamlit.io/cloud)
- Public app link: https://msadok-load-forecasting.streamlit.app/

## 📁 Project Structure

```
load-forecasting/
├── app/
│   └── streamlit_app.py
├── models/
│   ├── random_forest_model.joblib
│   └── feature_metadata.joblib
├── data/
│   └── weatherkit_plus_load.csv
├── src/
├── requirements.txt
└── README.md
```

## 📷 Preview

![App Screenshot](https://user-images.githubusercontent.com/placeholder/screenshot.png)

## 🔧 Setup Locally

```bash
git clone https://github.com/agastli/load-forecasting.git
cd load-forecasting
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## 🏗️ Future Improvements

- Add MAPE / RMSE evaluation
- Support live retraining
- Upload & compare multiple forecast runs

---

**Author:** Mohamed Sadok Gastli  
Built with 💡 by [CodeGPT](https://chatgpt.com/g/g-odWlfAKWM-lega)
