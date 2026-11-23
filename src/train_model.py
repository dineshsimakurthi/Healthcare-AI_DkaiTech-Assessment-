"""
PART 1 - MACHINE LEARNING
Predict diabetes (Outcome = 1) from the Pima Indians Diabetes dataset.

Steps (same order as the assignment):
  1. Load + clean the data
  2. Handle missing values
  3. Train / test split
  4. Train two classification models
  5. Report accuracy, precision, recall, F1, ROC-AUC
  6. Pick the best model
  7. Save the model (see predict.py for the prediction function)

Run:  python src/train_model.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "diabetes.csv"
MODEL_PATH = ROOT / "models" / "diabetes_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
TARGET = "Outcome"

# In this dataset a value of 0 here is impossible (nobody has 0 blood pressure),
# so 0 really means "missing".
ZERO_MEANS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


# ---------------------------------------------------------------- 1 & 2
def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    df = df.drop_duplicates()                                   # clean
    df[ZERO_MEANS_MISSING] = df[ZERO_MEANS_MISSING].replace(0, np.nan)  # mark missing

    print("\nMissing values per column:")
    print(df[ZERO_MEANS_MISSING].isna().sum().to_string())
    print(f"\nClass balance:\n{df[TARGET].value_counts(normalize=True).round(2).to_string()}")
    return df


# ---------------------------------------------------------------- 4
def build_models() -> dict:
    """Each model is a Pipeline: fill missing values -> scale -> classifier.
    Doing the imputing INSIDE the pipeline means it is learned from training
    data only (no data leakage from the test set)."""
    return {
        "Logistic Regression": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]),
        "Random Forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=200, max_depth=5,
                                             random_state=42)),
        ]),
    }


# ---------------------------------------------------------------- 5
def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def main():
    df = load_and_clean()
    X, y = df[FEATURES], df[TARGET]

    # ------------------------------------------------------------ 3
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"\nTrain: {len(X_train)} rows | Test: {len(X_test)} rows")

    results = {}
    trained = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        trained[name] = model
        results[name] = evaluate(model, X_test, y_test)

    table = pd.DataFrame(results).T.round(3)
    print("\n=== Test-set results ===")
    print(table.to_string())

    # ------------------------------------------------------------ 6
    # ROC-AUC is used to pick the winner because it does not depend on a
    # 0.5 threshold, and the classes are imbalanced (~35% diabetic).
    best_name = table["roc_auc"].idxmax()
    print(f"\nBest model by ROC-AUC: {best_name}")

    # ------------------------------------------------------------ 7
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(trained[best_name], MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(
        {"best_model": best_name, "metrics": table.to_dict(orient="index")}, indent=2))
    print(f"Saved model   -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
