import streamlit as st

login_user = st.sidebar.text_input("Username")

login_password = st.sidebar.text_input(
    "Password",
    type="password"
)

if login_user != "admin" or login_password != "admin123":
    st.warning("Please login to continue")
    st.stop()
    
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load model and columns
model = joblib.load("models/churn_model.pkl")
model_columns = joblib.load("models/model_columns.pkl")

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction System")

st.markdown(
    "### Predict whether a telecom customer is likely to churn using Machine Learning"
)

st.sidebar.title("About Project")

st.sidebar.info(
    '''
    This Machine Learning project predicts customer churn.

    Built using:
    - Python
    - Scikit-learn
    - Streamlit
    '''
)

st.write("---")
st.markdown(
    """
    <style>

    .main {
        background-color: #0E1117;
    }

    h1, h2, h3 {
        color: white;
    }

    .stButton button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-size: 18px;
        font-weight: bold;
    }

    .stMetric {
        background-color: #262730;
        padding: 10px;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Inputs
col1, col2 = st.columns(2)

with col1:

    SeniorCitizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    tenure = st.slider(
        "Tenure (months)",
        0,
        72,
        12
    )

with col2:

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

    probability = model.predict_proba(input_df)[0][1]

    st.write("---")

    st.subheader("Prediction Result")

    st.metric(
        label="Churn Probability",
        value=f"{probability:.2%}"
        )

        # Visualization Chart

    st.write("---")

    st.subheader("Prediction Visualization")

    labels = ["Stay", "Churn"]

    values = [1 - probability, probability]

    fig, ax = plt.subplots()

    ax.bar(labels, values)

    ax.set_ylabel("Probability")

    ax.set_title("Customer Churn Prediction")

    st.pyplot(fig)
    
    st.write("---")

st.subheader("Feature Importance")

try:

    importance = model.feature_importances_

    feature_importance_df = pd.DataFrame({
        "Feature": model_columns,
        "Importance": importance
    })

    feature_importance_df = feature_importance_df.sort_values(
        by="Importance",
        ascending=False
    ).head(10)

    fig2, ax2 = plt.subplots(figsize=(8, 5))

    ax2.barh(
        feature_importance_df["Feature"],
        feature_importance_df["Importance"]
    )

    ax2.set_xlabel("Importance")

    ax2.set_title("Top 10 Important Features")

    st.pyplot(fig2)

except:
    st.warning("Feature importance not available for this model.")