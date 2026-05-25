import streamlit as st
import pandas as pd
import joblib

# Load model and columns
model = joblib.load("models/churn_model.pkl")
model_columns = joblib.load("models/model_columns.pkl")

st.title("Customer Churn Prediction App")

st.write("Enter customer details below:")

# Inputs
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])

tenure = st.slider("Tenure (months)", 0, 72, 12)

MonthlyCharges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=500.0,
    value=50.0
)

TotalCharges = st.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=10000.0,
    value=1000.0
)

# Create input dictionary
input_data = {
    "SeniorCitizen": SeniorCitizen,
    "tenure": tenure,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges
}

# Convert to dataframe
input_df = pd.DataFrame([input_data])

# Add missing columns
for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Arrange columns
input_df = input_df[model_columns]

# Prediction button
if st.button("Predict Churn"):

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.error("Customer is likely to churn.")
    else:
        st.success("Customer is likely to stay.")