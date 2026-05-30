import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

def run_training_pipeline():
    print("🚀 Initializing Production Training Pipeline...")
    
    # 1. Mock Data Generation (Replace this block with your actual dataset loading)
    # e.g., df = pd.read_csv("data/telecom_churn.csv")
    np.random.seed(42)
    n_samples = 1000
    data = {
        'SeniorCitizen': np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2]),
        'tenure': np.random.randint(1, 72, size=n_samples),
        'MonthlyCharges': np.random.uniform(18.0, 120.0, size=n_samples),
        'TotalCharges': np.random.uniform(20.0, 8000.0, size=n_samples),
        'Churn': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    }
    df = pd.DataFrame(data)
    
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # Save the feature columns structure
    model_columns = list(X.columns)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model_columns, "models/model_columns.pkl")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Define Model Candidate Pool
    models = {
        "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    }
    
    performance_registry = {}
    best_f1 = 0
    best_model_name = None
    best_model_object = None
    
    # 3. Model Evaluation Loop
    for name, model in models.items():
        print(f"🔄 Training {name}...")
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        
        # Calculate production metrics
        metrics = {
            "Accuracy": round(accuracy_score(y_test, preds), 4),
            "Precision": round(precision_score(y_test, preds), 4),
            "Recall": round(recall_score(y_test, preds), 4),
            "F1_Score": round(f1_score(y_test, preds), 4),
            "ROC_AUC": round(roc_auc_score(y_test, probs), 4)
        }
        
        performance_registry[name] = metrics
        print(f"📊 {name} Metrics: {metrics}")
        
        # Selection logic based on F1-Score (balances precision and recall for churn)
        if metrics["F1_Score"] > best_f1:
            best_f1 = metrics["F1_Score"]
            best_model_name = name
            best_model_object = model

    print(f"🏆 Best Model Selected: {best_model_name} (F1: {best_f1})")
    
    # 4. Export Artifacts
    joblib.dump(best_model_object, "models/churn_model.pkl")
    with open("models/evaluation_metrics.json", "w") as f:
        json.dump(performance_registry, f, indent=4)
        
    print("💾 Artifacts saved successfully to /models/ folder.")

if __name__ == "__main__":
    run_training_pipeline()
