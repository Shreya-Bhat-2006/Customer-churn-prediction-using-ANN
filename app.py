import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
import joblib

# Load files
model = keras.models.load_model("churn_model.h5")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("📊 Customer Churn Prediction")

# ---------- USER INPUT ----------
tenure = st.slider("Tenure", 0, 72, 12)
monthly = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
total = st.number_input("Total Charges", 0.0, 10000.0, 500.0)

gender = st.selectbox("Gender", ["Male", "Female"])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
phone = st.selectbox("Phone Service", ["Yes", "No"])
multiple = st.selectbox("Multiple Lines", ["Yes", "No"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer", "Credit card"
])

# ---------- PREPROCESS ----------
def preprocess():
    data = {}

    # Numeric
    data["tenure"] = tenure
    data["MonthlyCharges"] = monthly
    data["TotalCharges"] = total

    # Binary
    data["gender"] = 1 if gender == "Female" else 0
    data["Partner"] = 1 if partner == "Yes" else 0
    data["Dependents"] = 1 if dependents == "Yes" else 0
    data["PhoneService"] = 1 if phone == "Yes" else 0
    data["MultipleLines"] = 1 if multiple == "Yes" else 0
    data["PaperlessBilling"] = 1 if paperless == "Yes" else 0

    # Initialize all columns = 0
    input_df = pd.DataFrame(columns=columns)
    input_df.loc[0] = 0

    # Fill values
    for key in data:
        input_df[key] = data[key]

    # One-hot encoding
    input_df[f"InternetService_{internet}"] = 1
    input_df[f"Contract_{contract}"] = 1
    input_df[f"PaymentMethod_{payment}"] = 1

    # Scale
    input_df[["tenure","MonthlyCharges","TotalCharges"]] = scaler.transform(
        input_df[["tenure","MonthlyCharges","TotalCharges"]]
    )

    return input_df

# ---------- PREDICT ----------
if st.button("Predict"):
    input_df = preprocess()
    prob = model.predict(input_df)[0][0]

    if prob > 0.4:
        st.error(f"⚠️ Customer likely to churn\nProbability: {prob:.2f}")
    else:
        st.success(f"✅ Customer will stay\nProbability: {prob:.2f}")