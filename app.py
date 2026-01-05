import streamlit as st
import pandas as pd
import joblib
import os
# load pretrained model 
st.title("   Daily Bike Rental Prediction")
st.write("Predict daily bike rental demand based on environmental and seasonal factors")
MODEL_PATHS = {
    "Final Model (Best)": "models/bike_rental_model_final.pkl",
    "Linear Regression": "models/Linear_Regression.pkl",
    "Random Forest": "models/Random_forest.pkl",
    "Support Vector Machine": "models/Support_vector_machine.pkl",
    "XGBoost": "models/xgboost.pkl"
}

selected_model = st.selectbox(
    "Select ML Model",
    list(MODEL_PATHS.keys()),
    help="Choose which trained model to use for prediction"
)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(MODEL_PATHS[selected_model])

st.caption(f"Model used: {selected_model}")
st.set_page_config(page_title="Bike Rental Prediction", layout="centered")

#User inputs
temp = st.slider("Temperature", 0.0, 1.0, 0.5)
atemp = st.slider("Feels Like Temperature", 0.0, 1.0, 0.5)
hum = st.slider("Humidity", 0.0, 1.0, 0.5)
windspeed = st.slider("Wind Speed", 0.0, 1.0, 0.2)

season = st.selectbox(
    "Season",
    ["Spring", "Summer", "Fall", "Winter"],
    help="Season of the year. Different seasons strongly influence bike rental demand."
)
season_mapping = {
    "Spring": 1,
    "Summer": 2,
    "Fall": 3,
    "Winter": 4
}
season = season_mapping[season]

#season = st.selectbox("Season", [1, 2, 3, 4])
#holiday = st.selectbox("Holiday", [0, 1])
holiday_label = st.selectbox(
    "Holiday",
    ["No", "Yes"],
    help="Indicates whether the day is a public holiday. Rentals are usually lower on holidays."
)
holiday = 1 if holiday_label == "Yes" else 0

weekday = st.selectbox("Weekday", [0, 1, 2, 3, 4, 5, 6])
workingday = st.selectbox("Working Day", [0, 1])
weathersit = st.selectbox("Weather Situation", [1, 2, 3])

year = st.selectbox("Year", [2011, 2012])
month = st.selectbox("Month", list(range(1, 13)))
day = st.selectbox("Day", list(range(1, 32)))

# ---- Prediction ----
if st.button("Predict Bike Rentals"):
    input_data = pd.DataFrame([{
        "temp": temp,
        "atemp": atemp,
        "hum": hum,
        "windspeed": windspeed,
        "year": year,
        "month": month,
        "day": day,
        "season": season,
        "holiday": holiday,
        "weekday": weekday,
        "workingday": workingday,
        "weathersit": weathersit
    }])

    prediction = model.predict(input_data)[0]
   

    st.success(f" Predicted Daily Bike Rentals: {round(float(prediction),2)}")
