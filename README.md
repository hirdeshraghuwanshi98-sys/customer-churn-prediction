# 📊 Enterprise Customer Churn Prediction Ecosystem

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/FastAPI-v0.110-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Dashboard](https://img.shields.io/badge/Streamlit-v1.45-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![Database](https://img.shields.io/badge/SQLite-3-003B57?style=for-the-badge&logo=sqlite)](https://www.sqlite.org/)

An end-to-end, enterprise-grade Machine Learning Engineering pipeline designed to ingest consumer telemetry metrics, benchmark multiple classification model architectures, and serve real-time subscriber attrition risk profiles via an interactive analytics dashboard, backed by a localized SQLite storage tier.

---

## 🚀 Live Production Links & Access
* **Interactive Frontend Dashboard:** [Streamlit Service UI](https://customer-churn-prediction-47zyecvht4xpvk8mninywq.streamlit.app/)

### 🔑 Demo Evaluation Credentials
To bypass the security authentication gateway on the live production interface, please utilize the following credentials:
* **Username:** `admin`
* **Password:** `admin123`

---

## 📊 Model Benchmarking & Evaluation Scoreboard

The core MLOps pipeline (`train.py`) executes automated competitive evaluation training across different model architectures. The champion model is selected dynamically using the **F1-Score** to balance business retention costs accurately.

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier** | **83.50%** | **74.12%** | **68.48%** | **71.19%** | **0.8840** | 🏆 **Champion Model** |
| Random Forest | 81.00% | 71.25% | 61.96% | 66.28% | 0.8560 | Baseline |

---

## 🛠️ Step-by-Step System Walkthrough

### Step 1: Automated Pipeline Training (`train.py`)
Running the local orchestration script triggers the evaluation engine. It loops through data profiles, extracts test weights, and generates data schemas along with performance logs inside the `/models/` directory.

<Image src="image_agent_tag_10519496926479358268" alt="Receiver Operating Characteristic ROC Curve plots benchmarking multiple classifier architectures side by side with heat map confusion matrices" caption="Automated Model Competitive Performance Benchmarking and Performance Metrics Generation" />

### Step 2: Operational Dashboard Analysis (`app.py`)
The Streamlit frontend loads the serialized champion artifacts seamlessly, rendering secure administrative controls, live metrics risk gauges, and feature importance matrices.

<Image src="image_agent_tag_1051949692647935121" alt="Streamlit cloud application user interface exhibiting customer subscription attrition analytics, risk score gauges, distribution plots, and charge parameters" caption="Production Streamlit Interactive Insights and Client Assessment Interface" />

---

## 📂 Repository Blueprint

```text
customer-churn-prediction/
│
├── train.py                  # Automated baseline model comparison & training engine
├── app.py                    # Multi-tab operational Streamlit analytics view
├── api.py                    # High-throughput FastAPI core inference gateway
│
├── data/
│   └── production_history.db # Integrated local SQLite telemetry tracking storage
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

## 🏎️ Local Orchestration Blueprint

Follow these exact operational steps to spin up the complete model environment, local relational logging tables, and application services on your machine:

### 1. Environment Alignment & Package Injection
Clone the target workspace repository and configure an isolated virtual runtime environment shell layout:
```bash
# Clone and enter the directory layout
git clone [https://github.com/your-username/customer-churn-prediction.git](https://github.com/your-username/customer-churn-prediction.git)
cd customer-churn-prediction

# Initialize and activate the local virtual shell environment
python -m venv .venv
source .venv/Scripts/activate  # Windows terminal command: .venv\Scripts\activate

# Clean download all explicitly pinned system package requirements
pip install -r requirements.txt

# Trains models and generates evaluation metadata
python train.py

#Deploy App Server Instances Locally
#To boot up the Interactive Frontend Streamlit Interface:
streamlit run app.py

#To initialize the high-speed Production FastAPI Inference Engine:
uvicorn api.py:app --reload --port 8000

📈 Future Optimization Roadmap
[ ] Integrate automated CI/CD staging test workflows via GitHub Actions configurations.
[ ] Implement advanced Explainable AI (XAI) feature weight maps utilizing SHAP core engine calculations.
[ ] Migrate local SQLite structured logging layouts into high-availability cloud cluster nodes (PostgreSQL / AWS RDS).
