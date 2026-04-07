import csv
import json
from pathlib import Path

import joblib
import numpy as np
import paho.mqtt.client as mqtt


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
FEATURE_ORDER_PATH = BASE_DIR / "feature_order.csv"
LABEL_MAP_PATH = BASE_DIR / "label_map.json"

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
TOPIC_FEATURES = "fl/features"
TOPIC_ALERTS = "fl/alerts"
TOPIC_PREDICTIONS = "fl/predictions"


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


model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
feature_names = load_feature_order()
label_names = load_label_names()


def infer(features):
    X = np.asarray(features, dtype=np.float32).reshape(1, -1)
    if X.shape[1] != len(feature_names):
        raise ValueError(f"Se esperaban {len(feature_names)} features y llegaron {X.shape[1]}")

    if scaler is not None:
        X = scaler.transform(X)

    pred = int(model.predict(X)[0])
    if hasattr(model, "predict_proba"):
        confidence = float(np.max(model.predict_proba(X)[0]))
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        confidence = float(np.max(softmax(decision[0] if decision.ndim > 1 else [0.0, decision[0]])))
    else:
        confidence = None

    return pred, confidence


def on_connect(client, userdata, flags, rc):
    print(f"[RPI] MQTT conectado rc={rc} | modelo=Random Forest")
    client.subscribe(TOPIC_FEATURES)


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        client_id = payload.get("client_id", "unknown")
        features = payload["features"]
        pred, confidence = infer(features)

        result = {
            "client_id": client_id,
            "model": "Random Forest",
            "predicted_class": pred,
            "predicted_label": label_names[pred],
            "confidence": confidence,
        }
        print("[RPI]", json.dumps(result))
        client.publish(TOPIC_PREDICTIONS, json.dumps(result))

        if pred != 0:
            alert = {
                "client_id": client_id,
                "alert": True,
                "attack_type": label_names[pred],
                "attack_probability": confidence,
                "model": "Random Forest",
            }
            client.publish(TOPIC_ALERTS, json.dumps(alert))
    except Exception as exc:
        print(f"[RPI][ERROR] {exc}")


if __name__ == "__main__":
    client = mqtt.Client(client_id="rpi_gateway_random_forest")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()
