# 🌧️ Rainfall Prediction App

A machine learning web app that predicts whether it will rain based on weather inputs like pressure, humidity, cloud cover, and more.

## 🔧 Tech Stack
- **Model:** Random Forest Classifier (tuned with GridSearchCV)
- **App:** Streamlit
- **Libraries:** scikit-learn, pandas, numpy

## 📥 Input Features
`pressure`, `dewpoint`, `humidity`, `cloud`, `sunshine`, `winddirection`, `windspeed`

## 🚀 Run Locally
```bash
git clone https://github.com/your-username/rainfall-prediction.git
cd rainfall-prediction
pip install streamlit pandas numpy scikit-learn
streamlit run app.py
```

## 📁 Files
| File | Description |
|------|-------------|
| `app.py` | Streamlit web app |
| `rainfall_prediction_model.pkl` | Trained model |
| `Rain_predict.ipynb` | EDA & model training notebook |

---
