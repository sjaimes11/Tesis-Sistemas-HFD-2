import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
FEATURE_ORDER_PATH = BASE_DIR / "feature_order.csv"
LABEL_MAP_PATH = BASE_DIR / "label_map.json"


def load_feature_order():
    with FEATURE_ORDER_PATH.open("r", encoding="utf-8", newline="") as handle:
        values = [row[0].strip() for row in csv.reader(handle) if row]
    return [value for value in values if value and value != "feature"]


def load_label_names():
    data = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
    return [data[str(index)] for index in sorted((int(key) for key in data.keys()))]


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    exps = np.exp(x)
    return exps / np.sum(exps)


def main():
    parser = argparse.ArgumentParser(description="Random Forest predictor para Raspberry Pi")
    parser.add_argument("--features-json", required=True, help="Lista JSON con las 13 features")
    args = parser.parse_args()

    features = np.asarray(json.loads(args.features_json), dtype=np.float32).reshape(1, -1)
    feature_names = load_feature_order()
    label_names = load_label_names()

    if features.shape[1] != len(feature_names):
        raise ValueError(f"Se esperaban {len(feature_names)} features y llegaron {features.shape[1]}")

    model = joblib.load(MODEL_PATH)
    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)
        features = scaler.transform(features)

    pred = int(model.predict(features)[0])
    if hasattr(model, "predict_proba"):
        confidence = float(np.max(model.predict_proba(features)[0]))
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(features)
        confidence = float(np.max(softmax(decision[0] if decision.ndim > 1 else [0.0, decision[0]])))
    else:
        confidence = None

    result = {
        "predicted_class": pred,
        "predicted_label": label_names[pred],
        "confidence": confidence,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
