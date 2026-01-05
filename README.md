# Bike Rental Demand Prediction (End-to-End ML Project)

## **Project Overview**

This project focuses on predicting daily bike rental demand based on environmental and seasonal factors such as weather conditions, 
temperature, humidity, wind speed, and calendar-related variables.
The solution shows a complete end-to-end machine learning workflow, from data analysis and modeling to real-time deployment using Streamlit.
---
## **Objective**

**To build and deploy a machine learning model that can accurately predict the total daily bike rental count (cnt), helping bike-sharing businesses with:**

 * Demand forecasting

 * Resource planning

## **Dataset**

Source: [click Here](https://d3ilbtxij3aepc.cloudfront.net/projects/CDS-Capstone-Projects/PRCP-1018-BikeRental.zip)

**Dataset Used: df_day (daily aggregated data)**

**Target Variable: cnt – total daily bike rentals**
---
## *Why df_day?*

   * Aligns with daily business decision-making
   * Reduces noise compared to hourly data
   * Captures overall demand trends effectively
---

## **Features Used**

 ###* Numerical Features
    * temp – Temperature
    * atemp – Feels-like temperature
    * hum – Humidity
    * windspeed – Wind speed
    * year, month, day

### **Categorical Features**

  * season
  * holiday
  * weekday
  * workingday
  * weathersit
---
### **Data Preprocessing**

*Used ColumnTransformer for handling mixed data types*
*Numerical features passed through without scaling*
*Categorical features encoded using OneHotEncoder*
*Preprocessing included inside a Pipeline to prevent data leakage*
--- 
### **Models Trained & Compared**
  * Linear Regression (Baseline)
  * Decision Tree Regressor
  * Random Forest Regressor (Bagging)
  * Support Vector Machine (SVR)**
  * XGBoost Regressor (Boosting)
---
### **Model Selection**

**GridSearchCV with 5-fold cross-validation used for hyperparameter tuning**

* *Final model selected based on R², MAE, and RMSE*

#### **Evaluation Metrics**

   * R² Score – Variance explained by the model
   * MAE – Average absolute error
   * RMSE – Penalizes large prediction errors
---
## Model Deployment (Streamlit)
  * The trained models were deployed using Streamlit, enabling:
  * Real-time prediction of daily bike rentals
  * User-friendly inputs (Yes/No labels, season names, weather descriptions)

 *Model selection (switch between multiple trained models)*


## **Deployment Highlights**

 * Trained models saved as .pkl files
 * Dynamic model loading in Streamlit

---
## Project Structure
bike_rental_demand/
│
├── app.py                  # Streamlit application
├── models/                 # Saved ML models (.pkl)
│   ├── bike_rental_model_final.pkl
│   ├── Random_forest.pkl
│   ├── Support_vector_machine.pkl
│   └── xgboost.pkl
│
├── requirements.txt
├── README.md
├── .gitignore
---

## Installation & Run Locally

 * Create virtual environment
   `python -m venv venv`

 * Activate venv (Windows)
  `venv\Scripts\activate`

 * Install dependencies
`pip install -r requirements.txt`

  * Run Streamlit app
`streamlit run app.py`

---
## requirements.txt
streamlit
pandas
numpy
scikit-learn
joblib
xgboost
---

## **Limitations**
 * Daily aggregation hides intra-day demand patterns
 * Limited to weather and calendar features
 * Rare weather events are underrepresented
 * No explicit time-series modeling

## **Future Work**

* **Add lag features and rolling averages**
* **Use hourly data for real-time demand forecasting**
* **Deploy using Docker / FastAPI
---
# Author
Mohd Huzaif
Machine Learning / Data Science Enthusiast
