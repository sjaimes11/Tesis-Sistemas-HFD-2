import base64
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
import paho.mqtt.client as mqtt
import requests

from ascon128 import encrypt as ascon_encrypt, decrypt as ascon_decrypt, generate_nonce
from ascon_metrics import AsconMetrics


BASE_DIR = Path(__file__).resolve().parent
RPI_DIR = BASE_DIR / "raspberry"
MODEL_PATH = RPI_DIR / "model.pkl"
SCALER_PATH = RPI_DIR / "scaler.pkl"
FEATURE_ORDER_PATH = RPI_DIR / "feature_order.csv"
LABEL_MAP_PATH = RPI_DIR / "label_map.json"

GATEWAY_ID = "gateway_svm"
MQTT_LOCAL_BROKER = "localhost"
MQTT_LOCAL_PORT = 1883
TOPIC_FEATURES = "fl/features"
TOPIC_ALERTS = "fl/alerts"
TOPIC_PREDICTIONS = "fl/predictions"
SERVER_URL = "http://192.168.40.95:8001/ingest-prediction"

ASCON_KEY = bytes([0xA1, 0xB2, 0xC3, 0xD4, 0xE5, 0xF6, 0x07, 0x18,
                   0x29, 0x3A, 0x4B, 0x5C, 0x6D, 0x7E, 0x8F, 0x90])
MSG_COUNTER = 0

metrics = AsconMetrics("gateway_svm")


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


def decrypt_payload(raw_payload, direction):
    envelope = json.loads(raw_payload.decode("utf-8"))
    ct = base64.b64decode(envelope["ct"])
    tag = base64.b64decode(envelope["tag"])
    nonce = base64.b64decode(envelope["nonce"])

    t0 = time.perf_counter()
    plaintext = ascon_decrypt(ct, ASCON_KEY, nonce, tag)
    dec_ms = (time.perf_counter() - t0) * 1000
    if plaintext is None:
        return None

    metrics.record(direction, "decrypt", len(plaintext), len(raw_payload), dec_ms, 0)
    return json.loads(plaintext.decode("utf-8"))


def encrypt_payload(payload_dict, direction):
    global MSG_COUNTER
    payload_bytes = json.dumps(payload_dict).encode("utf-8")
    nonce = generate_nonce(int(time.time() * 1000), MSG_COUNTER)
    MSG_COUNTER += 1

    t0 = time.perf_counter()
    ciphertext, tag = ascon_encrypt(payload_bytes, ASCON_KEY, nonce)
    enc_ms = (time.perf_counter() - t0) * 1000

    envelope = {
        "ct": base64.b64encode(ciphertext).decode("ascii"),
        "tag": base64.b64encode(tag).decode("ascii"),
        "nonce": base64.b64encode(nonce).decode("ascii"),
    }
    metrics.record(direction, "encrypt", len(payload_bytes), len(json.dumps(envelope)), enc_ms, 0)
    return envelope


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


def forward_to_server(result):
    envelope = encrypt_payload(result, "RPi->PC")
    try:
        requests.post(SERVER_URL, json=envelope, timeout=5)
    except Exception as exc:
        print(f"[GATEWAY][WARN] No se pudo enviar al server: {exc}")


def on_connect(client, userdata, flags, rc):
    print(f"[GATEWAY] MQTT conectado rc={rc} | modelo=SVM")
    client.subscribe(TOPIC_FEATURES)


def on_message(client, userdata, msg):
    try:
        payload = decrypt_payload(msg.payload, "ESP32->RPi")
        if payload is None:
            print("[GATEWAY][ERROR] Tag ASCON inválido.")
            return

        client_id = payload.get("client_id", "unknown")
        features = payload["features"]
        pred, confidence = infer(features)

        result = {
            "gateway_id": GATEWAY_ID,
            "client_id": client_id,
            "model": "SVM",
            "predicted_class": pred,
            "predicted_label": label_names[pred],
            "confidence": confidence,
        }
        print("[GATEWAY]", json.dumps(result))
        client.publish(TOPIC_PREDICTIONS, json.dumps(result))

        if pred != 0:
            alert = {
                "client_id": client_id,
                "alert": True,
                "attack_type": label_names[pred],
                "attack_probability": confidence,
                "model": "SVM",
            }
            client.publish(TOPIC_ALERTS, json.dumps(alert))

        forward_to_server(result)
    except Exception as exc:
        print(f"[GATEWAY][ERROR] {exc}")


if __name__ == "__main__":
    client = mqtt.Client(client_id="gateway_hfl_fog_svm")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_LOCAL_BROKER, MQTT_LOCAL_PORT, 60)
    client.loop_forever()
