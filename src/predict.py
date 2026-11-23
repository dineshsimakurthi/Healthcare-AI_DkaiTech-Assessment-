"""Prediction function for the saved diabetes model."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "diabetes_model.joblib"
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

_model = None  # loaded once, reused


def predict_risk(patient: dict) -> dict:
    """patient example:
    {"Pregnancies": 2, "Glucose": 150, "BloodPressure": 80, "SkinThickness": 30,
     "Insulin": 120, "BMI": 32.5, "DiabetesPedigreeFunction": 0.5, "Age": 45}
    Use None (or 0) for a value that is unknown - the model fills it in.
    """
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)

    row = {f: patient.get(f) for f in FEATURES}
    # Same rule as training: 0 in these columns means "missing"
    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        if row[col] in (0, None):
            row[col] = np.nan

    prob = float(_model.predict_proba(pd.DataFrame([row], columns=FEATURES))[0, 1])
    level = "High" if prob >= 0.6 else "Moderate" if prob >= 0.35 else "Low"
    return {"probability": round(prob, 3), "risk_level": level,
            "prediction": "Diabetic" if prob >= 0.5 else "Non-diabetic"}


if __name__ == "__main__":
    print(predict_risk({"Pregnancies": 2, "Glucose": 150, "BloodPressure": 80,
                        "SkinThickness": 30, "Insulin": 120, "BMI": 32.5,
                        "DiabetesPedigreeFunction": 0.5, "Age": 45}))
