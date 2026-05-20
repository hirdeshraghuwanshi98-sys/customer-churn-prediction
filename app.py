import streamlit as st
import pandas as pd
import joblib

# Load saved model and feature columns
model = joblib.load("models/churn_model.pkl")
model_columns = joblib.load("models/model_columns.pkl")

# Page title
st.title("Customer Churn Prediction")
st.write("Enter customer details to predict whether the customer will churn.")

# Input fields
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=840.0)

# Prediction button
if st.button("Predict Churn"):

    # Create dictionary with all model columns initialized to 0
    input_data = {col: 0 for col in model_columns}

    # Fill numerical features
    if "tenure" in input_data:
        input_data["tenure"] = tenure
    if "MonthlyCharges" in input_data:
        input_data["MonthlyCharges"] = monthly_charges
    if "TotalCharges" in input_data:
        input_data["TotalCharges"] = total_charges

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Show result
    if prediction == 1:
        st.error(f"Customer is likely to churn. Probability: {probability:.2%}")
    else:
        st.success(f"Customer is likely to stay. Churn probability: {probability:.2%}")