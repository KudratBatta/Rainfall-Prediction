import streamlit as st
import pandas as pd
import pickle
import numpy as np
import streamlit as st
import base64

def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        
        </style>
        """,
        unsafe_allow_html=True
    )

# Call function
set_background("background.jpg")

# Page config
st.set_page_config(
    page_title="Rainfall Prediction App",
    page_icon="🌧️",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    with open("rainfall_prediction_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    return model_data["model"], model_data["feature_names"]

model, feature_names = load_model()

# UI
st.markdown(
    """
    <h1 style='text-align: center;'>🌧️ Rainfall Prediction App</h1>
    <p style='text-align: center; font-size:18px;'>
        Enter the weather parameters below to predict whether it will rain.
    </p>
    """,
    unsafe_allow_html=True
)
st.divider()

col1, col2 = st.columns(2)

with col1:
    pressure = st.number_input(
        "Pressure (hPa)",
        min_value=900.0, max_value=1100.0,
        value=1015.9, step=0.1,
        help="Atmospheric pressure in hPa"
    )
    dewpoint = st.number_input(
        "Dew Point (°C)",
        min_value=-20.0, max_value=40.0,
        value=19.9, step=0.1,
        help="Dew point temperature"
    )
    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0, max_value=100.0,
        value=95.0, step=1.0,
        help="Relative humidity percentage"
    )
    cloud = st.number_input(
        "Cloud Cover (%)",
        min_value=0.0, max_value=100.0,
        value=81.0, step=1.0,
        help="Cloud cover percentage"
    )

with col2:
    sunshine = st.number_input(
        "Sunshine (hours)",
        min_value=0.0, max_value=24.0,
        value=0.0, step=0.1,
        help="Sunshine duration in hours"
    )
    winddirection = st.number_input(
        "Wind Direction (°)",
        min_value=0.0, max_value=360.0,
        value=40.0, step=1.0,
        help="Wind direction in degrees (0-360)"
    )
    windspeed = st.number_input(
        "Wind Speed (km/h)",
        min_value=0.0, max_value=200.0,
        value=13.7, step=0.1,
        help="Wind speed in km/h"
    )

st.divider()

if st.button("🔍 Predict Rainfall", use_container_width=True, type="primary"):
    input_data = (pressure, dewpoint, humidity, cloud, sunshine, winddirection, windspeed)
    input_df = pd.DataFrame([input_data], columns=feature_names)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    st.markdown("### Prediction Result")

    if prediction == 1:
        st.error("🌧️ **Rainfall Expected!**")
        rain_prob = probability[1] * 100
        st.metric("Rainfall Probability", f"{rain_prob:.1f}%")
    else:
        st.success("☀️ **No Rainfall Expected**")
        no_rain_prob = probability[0] * 100
        st.metric("No Rainfall Probability", f"{no_rain_prob:.1f}%")

    # Probability breakdown
    st.markdown("#### Probability Breakdown")
    prob_df = pd.DataFrame({
        "Outcome": ["No Rainfall ☀️", "Rainfall 🌧️"],
        "Probability (%)": [round(probability[0] * 100, 2), round(probability[1] * 100, 2)]
    })
    st.dataframe(prob_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Model: Random Forest Classifier | Trained on historical rainfall data")
