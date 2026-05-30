import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
import os

st.set_page_config(
    page_title="Enterprise Churn Analytics",
    page_icon="🔮",
    layout="wide"
)

# 1. Production Security Settings
ADMIN_USER = st.secrets.get("ADMIN_USER", "admin")
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "admin123")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.sidebar.title("🔐 Production Gateway")
    user = st.sidebar.text_input("Username")
    pas = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if user == ADMIN_USER and pas == ADMIN_PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.sidebar.error("Invalid Credentials")
    st.warning("Please sign in to access the secure predictive layer.")
    st.stop()

# 2. Cached Resource Loader
@st.cache_resource
def load_assets():
    mdl = joblib.load("models/churn_model.pkl")
    cols = joblib.load("models/model_columns.pkl")
    metrics = {}
    if os.path.exists("models/evaluation_metrics.json"):
        with open("models/evaluation_metrics.json", "r") as f:
            metrics = json.load(f)
    return mdl, cols, metrics

try:
    model, model_columns, evaluation_metrics = load_assets()
except Exception as e:
    st.error(f"Failed to load production models. Ensure you have run `train.py` first. Error: {e}")
    st.stop()

# 3. Main Dashboard Layout Split
tabs = st.tabs(["🎯 Real-Time Prediction Engine", "📊 Model Comparison & Governance"])

# --- TAB 1: PREDICTIONS ---
with tabs[0]:
    st.title("🔮 Customer Churn Prediction Engine")
    st.markdown("Enter telemetry data to assess real-time attrition probabilities.")
    st.write("---")
    
    col1, col2 = st.columns(2)
    with col1:
        SeniorCitizen = st.selectbox("Senior Citizen Status", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        tenure = st.slider("Tenure Matrix (Months)", 0, 72, 24)
    with col2:
        MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=65.0)
        TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=1500.0)

    input_df = pd.DataFrame([{
        "SeniorCitizen": SeniorCitizen,
        "tenure": tenure,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }])
    
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_columns]

    if st.button("Execute Risk Evaluation", type="primary"):
        prob = model.predict_proba(input_df)[0][1]
        
        st.write("---")
        metric_col1, metric_col2 = st.columns([1, 2])
        
        with metric_col1:
            st.metric(label="Calculated Attrition Risk", value=f"{prob:.2%}")
            if prob >= 0.5:
                st.error("🚨 HIGH RISK PROFILE DETECTED")
            else:
                st.success("✅ LOW RISK RETAINED PROFILE")
                
        with metric_col2:
            fig, ax = plt.subplots(figsize=(6, 2))
            colors = ['#2ecc71', '#e74c3c']
            ax.barh(["Stay Risk", "Churn Risk"], [1-prob, prob], color=colors)
            ax.set_xlim(0, 1)
            st.pyplot(fig)

# --- TAB 2: MODEL GOVERNANCE ---
with tabs[1]:
    st.title("📊 Model Benchmarking & Metric Logs")
    st.markdown("This section reviews audit trails and evaluation performance criteria tracked across models.")
    st.write("---")
    
    if evaluation_metrics:
        # Transform JSON into a clear display DataFrame
        metrics_df = pd.DataFrame(evaluation_metrics).T
        st.subheader("📋 Leaderboard Matrix")
        st.dataframe(metrics_df.style.highlight_max(axis=0, color="#1e3d2f"))
        
        # Performance Comparison Plot
        st.subheader("📈 Visual Comparison Matrix")
        fig_comp, ax_comp = plt.subplots(figsize=(10, 4))
        metrics_df.plot(kind="bar", ax=ax_comp)
        plt.title("Performance Benchmarks Across Structural Models")
        plt.ylabel("Score Range (0-1)")
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig_comp)
    else:
        st.info("No comparative tracking logs detected. Run `train.py` to populate pipeline matrices.")
