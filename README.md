# 📊 Enterprise Customer Churn Prediction Ecosystem

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/FastAPI-v0.110-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Dashboard](https://img.shields.io/badge/Streamlit-v1.45-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![Model](https://img.shields.io/badge/XGBoost-Enabled-0172B2?style=for-the-badge&logo=xgboost)](https://xgboost.readthedocs.io/)

An end-to-end, enterprise-grade Machine Learning Engineering pipeline designed to ingest consumer telemetry metrics, benchmark multiple classification model architectures, and serve real-time subscriber attrition risk profiles via an interactive analytics dashboard and a high-throughput REST API.

---

## 🚀 Live Production Links & Access
* **Interactive Frontend Dashboard:** [Streamlit Service UI](https://customer-churn-prediction-47zyecvht4xpvk8mninywq.streamlit.app/)

### 🔑 Demo Evaluation Credentials
To bypass the security authentication gateway on the live production interface, please utilize the following credentials:
* **Username:** `admin`
* **Password:** `admin123`

---

## 📌 Project Architecture & Core Features

* **Multi-Model Competitive Benchmarking (`train.py`):** Dynamically trains and validates `RandomForest` and `XGBoost` architectures, tracking Precision, Recall, F1-Score, and ROC-AUC metrics to ensure mathematical alignment.
* **Auto-Artifact Serialization Engine:** Automatically serializes the optimal champion model based on the highest F1-Score to balance consumer retention economics.
* **Secure Production Client Interface (`app.py`):** Interactive UI featuring a Role-Based Authentication Gateway (Streamlit Secrets Layer), responsive prediction gauges, dynamic feature weight tracking, and performance comparison leaderboards.
* **High-Performance Restful Interface (`api.py`):** Production-ready FastAPI routing optimized with strict `Pydantic` schema validations and automated local log rotation tracking (`logs/production.log`).

---

## 🛠️ Tech Stack & Dependencies

* **Core Language:** Python 3.11
* **Machine Learning Pipelines:** Scikit-Learn, XGBoost
* **Data Processing & Analytics:** Pandas, NumPy
* **Visualization Layer:** Matplotlib
* **API Routing & Serving:** FastAPI, Uvicorn
* **Deployment Stack:** Streamlit Community Cloud, Docker

---

## 📂 Repository Blueprint

```text
customer-churn-prediction/
│
├── train.py                  # Automated baseline model comparison & training engine
├── app.py                    # Multi-tab operational Streamlit analytics view
├── api.py                    # High-throughput FastAPI core inference gateway
│
├── models/                   # Serialized pipeline assets directory
│   ├── churn_model.pkl       # Automated top-performing champion model
│   ├── model_columns.pkl     # Persisted validation structural shape arrays
│   └── evaluation_metrics.json  # Exported evaluation leaderboard metrics matrix
│
├── logs/
│   └── production.log        # Self-contained active execution error ledger
│
├── Dockerfile                # Multi-stage microservice image container context
├── requirements.txt          # Explicitly pinned application package distributions
└── README.md                 # Interactive architectural summary documentation
