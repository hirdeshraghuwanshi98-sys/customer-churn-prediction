import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)
from xgboost import XGBClassifier


def load_telco_dataset():
    """
    Load real IBM Telco Customer Churn dataset (7,043 rows, 20 features).
    Downloads directly from public source — no manual download needed.
    """
    print("📥 Loading IBM Telco Customer Churn Dataset (7,043 rows)...")

    url = (
        "https://raw.githubusercontent.com/"
        "IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    )

    try:
        df = pd.read_csv(url)
        print(f"✅ Dataset loaded successfully — {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        raise RuntimeError(
            f"❌ Failed to download dataset. Check internet connection.\nError: {e}"
        )

    return df


def preprocess(df):
    """
    Full preprocessing pipeline on real Telco dataset:
    - Fix TotalCharges dtype
    - Drop customerID (non-predictive)
    - Encode binary + multi-class categorical features
    - Encode target label
    """
    print("🔧 Running preprocessing pipeline...")

    # Fix TotalCharges — loaded as string due to empty spaces
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Drop non-predictive identifier
    df.drop(columns=["customerID"], inplace=True, errors="ignore")

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Encode binary categoricals (Yes/No/Male/Female etc.)
    binary_cols = [
        col for col in df.select_dtypes(include="object").columns
        if df[col].nunique() == 2
    ]
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])

    # One-hot encode remaining multi-class categoricals
    df = pd.get_dummies(df, drop_first=True)

    print(f"✅ Preprocessing complete — {df.shape[1] - 1} features, {df.shape[0]} samples")
    return df


def run_training_pipeline():
    print("🚀 Initializing Production Training Pipeline...")
    print("=" * 55)

    # 1. Load real dataset
    df = load_telco_dataset()

    # 2. Preprocess
    df = preprocess(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # 3. Save feature column structure for inference alignment
    model_columns = list(X.columns)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model_columns, "models/model_columns.pkl")
    print(f"💾 Saved model_columns.pkl — {len(model_columns)} features")

    # 4. Train/test split — stratified to preserve churn ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"📊 Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    print(f"📊 Churn rate in test set: {y_test.mean():.2%}")
    print("=" * 55)

    # 5. Model candidate pool
    candidates = {
        "Random_Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
            verbosity=0
        )
    }

    performance_registry = {}
    best_f1 = 0
    best_model_name = None
    best_model_object = None

    # 6. Training + evaluation loop
    for name, model in candidates.items():
        print(f"\n🔄 Training {name} on {len(X_train)} real Telco samples...")
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        metrics = {
            "Accuracy":  round(accuracy_score(y_test, preds),          4),
            "Precision": round(precision_score(y_test, preds,
                                               zero_division=0),        4),
            "Recall":    round(recall_score(y_test, preds,
                                            zero_division=0),           4),
            "F1_Score":  round(f1_score(y_test, preds,
                                        zero_division=0),               4),
            "ROC_AUC":   round(roc_auc_score(y_test, probs),            4)
        }

        performance_registry[name] = metrics

        print(f"📊 {name} Results:")
        print(f"   Accuracy  : {metrics['Accuracy']  * 100:.2f}%")
        print(f"   Precision : {metrics['Precision'] * 100:.2f}%")
        print(f"   Recall    : {metrics['Recall']    * 100:.2f}%")
        print(f"   F1-Score  : {metrics['F1_Score']  * 100:.2f}%")
        print(f"   ROC-AUC   : {metrics['ROC_AUC']:.4f}")

        # Champion selection by F1 (best metric for imbalanced churn data)
        if metrics["F1_Score"] > best_f1:
            best_f1          = metrics["F1_Score"]
            best_model_name  = name
            best_model_object = model

    # 7. Export champion model + metrics
    print("\n" + "=" * 55)
    print(f"🏆 Champion Model : {best_model_name}")
    print(f"   Best F1-Score  : {best_f1 * 100:.2f}%")

    joblib.dump(best_model_object, "models/churn_model.pkl")

    with open("models/evaluation_metrics.json", "w") as f:
        json.dump(performance_registry, f, indent=4)

    print("💾 Artifacts saved:")
    print("   models/churn_model.pkl")
    print("   models/model_columns.pkl")
    print("   models/evaluation_metrics.json")
    print("=" * 55)
    print("✅ Production training pipeline complete.")


if __name__ == "__main__":
    run_training_pipeline()
