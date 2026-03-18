"""
=============================================================================
 aggregator_hfl.py — Raspberry Pi (Fog Aggregator) — HFL v6 (3 clases)
=============================================================================
 Modelo: 13 -> 32 -> 16 -> 8 -> 3 (softmax)
 Capas FL: W3(16,8) + b3(8) + W4(8,3) + b4(3)
 
 Ejecutar: python3 aggregator_hfl.py
=============================================================================
 Requiere: pip3 install paho-mqtt numpy requests
 Mosquitto: sudo systemctl start mosquitto
=============================================================================
"""
import numpy as np
import requests
import json
import threading
import time
import paho.mqtt.client as mqtt
from http.server import HTTPServer, BaseHTTPRequestHandler

# ====================== ARQUITECTURA ======================
L2_UNITS = 16
L3_UNITS = 8
OUTPUT_UNITS = 3  # <-- 3 clases

CLASS_NAMES = ["normal", "mqtt_bruteforce", "scan_A"]

# ====================== MODELO GLOBAL ======================
W3_global = np.zeros((L2_UNITS, L3_UNITS), dtype=np.float32)
b3_global = np.zeros(L3_UNITS, dtype=np.float32)
W4_global = np.zeros((L3_UNITS, OUTPUT_UNITS), dtype=np.float32)
b4_global = np.zeros(OUTPUT_UNITS, dtype=np.float32)

# ====================== ACUMULADORES FEDAVG ======================
W3_update_sum = np.zeros((L2_UNITS, L3_UNITS), dtype=np.float32)
b3_update_sum = np.zeros(L3_UNITS, dtype=np.float32)
W4_update_sum = np.zeros((L3_UNITS, OUTPUT_UNITS), dtype=np.float32)
b4_update_sum = np.zeros(OUTPUT_UNITS, dtype=np.float32)
total_samples_this_round = 0
updates_received = 0
current_round = 0

MIN_UPDATES_PER_ROUND = 3

# ====================== CONFIGURACION ======================
IP_PC = "192.168.1.22"
PORT_PC = "8001"
url_coordinador = f"http://{IP_PC}:{PORT_PC}/aggregate-from-pi"

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
TOPIC_UPDATES = "fl/updates"
TOPIC_GLOBAL_MODEL = "fl/global_model"

mqtt_client = None


# ====================== HTTP SERVER ======================

class DeployModelHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        global W3_global, b3_global, W4_global, b4_global, current_round
        global W3_update_sum, b3_update_sum, W4_update_sum, b4_update_sum
        global total_samples_this_round, updates_received

        if self.path == "/deploy-model":
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            W3_global = np.array(data["W3"], dtype=np.float32)
            b3_global = np.array(data["b3"], dtype=np.float32)
            W4_global = np.array(data["W4"], dtype=np.float32)
            b4_global = np.array(data["b4"], dtype=np.float32)
            current_round = data.get("round", current_round + 1)

            W3_update_sum.fill(0); b3_update_sum.fill(0)
            W4_update_sum.fill(0); b4_update_sum.fill(0)
            total_samples_this_round = 0; updates_received = 0

            print(f"\n{'='*60}")
            print(f" MODELO GLOBAL 3-CLASS RECIBIDO DEL PC (Ronda {current_round})")
            print(f" W3 mag: {np.mean(np.abs(W3_global)):.6f}")
            print(f" W4 mag: {np.mean(np.abs(W4_global)):.6f}")
            print(f" W4 shape: {W4_global.shape}")
            print(f"{'='*60}")

            broadcast_model_to_esp32s()

            resp = json.dumps({"status": "ok", "round": current_round})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(resp.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def broadcast_model_to_esp32s():
    global mqtt_client

    payload = {
        "round": current_round,
        "W3": W3_global.tolist(),
        "b3": b3_global.tolist(),
        "W4": W4_global.tolist(),
        "b4": b4_global.tolist()
    }
    payload_str = json.dumps(payload)

    if mqtt_client and mqtt_client.is_connected():
        mqtt_client.publish(TOPIC_GLOBAL_MODEL, payload_str, qos=1)
        print(f"[MQTT] Modelo 3-class publicado ({len(payload_str)} bytes)")
    else:
        print("[ERROR] MQTT no conectado")


# ====================== MQTT CALLBACKS ======================

def on_connect(client, userdata, flags, rc):
    print(f"\n[MQTT] Conectado a Mosquitto (codigo: {rc})")
    client.subscribe(TOPIC_UPDATES)
    print(f"[MQTT] Suscrito a '{TOPIC_UPDATES}'\n")


def on_message(client, userdata, msg):
    global W3_update_sum, b3_update_sum, W4_update_sum, b4_update_sum
    global total_samples_this_round, updates_received

    try:
        data = json.loads(msg.payload.decode('utf-8'))
        client_id = data.get("client_id", "unknown")
        num_samples = data.get("num_samples", 0)
        round_num = data.get("round", 0)
        arch = data.get("model_arch", "?")

        print(f"[MQTT] Delta de '{client_id}' | Ronda: {round_num} | Muestras: {num_samples} | Arch: {arch}")

        weight_delta = data.get("weight_delta", {})
        delta_W3 = np.array(weight_delta.get("W3", []), dtype=np.float32)
        delta_b3 = np.array(weight_delta.get("b3", []), dtype=np.float32)
        delta_W4 = np.array(weight_delta.get("W4", []), dtype=np.float32)
        delta_b4 = np.array(weight_delta.get("b4", []), dtype=np.float32)

        # Validar shapes
        ok = True
        if delta_W3.shape != (L2_UNITS, L3_UNITS):
            print(f"  ERROR W3: {delta_W3.shape} != ({L2_UNITS},{L3_UNITS})"); ok = False
        if delta_b3.shape != (L3_UNITS,):
            print(f"  ERROR b3: {delta_b3.shape}"); ok = False
        if delta_W4.shape != (L3_UNITS, OUTPUT_UNITS):
            print(f"  ERROR W4: {delta_W4.shape} != ({L3_UNITS},{OUTPUT_UNITS})"); ok = False
        if delta_b4.shape != (OUTPUT_UNITS,):
            print(f"  ERROR b4: {delta_b4.shape}"); ok = False
        if not ok:
            return

        W3_update_sum += delta_W3 * num_samples
        b3_update_sum += delta_b3 * num_samples
        W4_update_sum += delta_W4 * num_samples
        b4_update_sum += delta_b4 * num_samples
        total_samples_this_round += num_samples
        updates_received += 1

        print(f"  Acumulado: {updates_received}/{MIN_UPDATES_PER_ROUND}")

        if updates_received >= MIN_UPDATES_PER_ROUND:
            do_fedavg_and_send()

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback; traceback.print_exc()


def do_fedavg_and_send():
    global W3_global, b3_global, W4_global, b4_global
    global W3_update_sum, b3_update_sum, W4_update_sum, b4_update_sum
    global total_samples_this_round, updates_received

    print(f"\n{'='*60}")
    print(f" FEDAVG 3-CLASS - {updates_received} updates, {total_samples_this_round} muestras")
    print(f"{'='*60}")

    W3_global = W3_update_sum / total_samples_this_round
    b3_global = b3_update_sum / total_samples_this_round
    W4_global = W4_update_sum / total_samples_this_round
    b4_global = b4_update_sum / total_samples_this_round

    W3_update_sum.fill(0); b3_update_sum.fill(0)
    W4_update_sum.fill(0); b4_update_sum.fill(0)
    total_samples_this_round = 0; updates_received = 0

    print(f"  W3 mag: {np.mean(np.abs(W3_global)):.6f}")
    print(f"  W4 mag: {np.mean(np.abs(W4_global)):.6f}")

    payload = {
        "round": current_round,
        "model": "3class_13_32_16_8_3",
        "classes": CLASS_NAMES,
        "W3": W3_global.tolist(),
        "b3": b3_global.tolist(),
        "W4": W4_global.tolist(),
        "b4": b4_global.tolist()
    }

    try:
        print(f"\n  Enviando FedAvg al PC -> POST {url_coordinador}")
        resp = requests.post(url_coordinador, json=payload, timeout=5)
        print(f"  Respuesta PC: {resp.json()}\n")
    except Exception as e:
        print(f"  ERROR contactando PC: {e}\n")


# ====================== MAIN ======================

if __name__ == "__main__":
    print("=" * 60)
    print(" AGREGADOR HFL v6 - 3-CLASS")
    print(f" Modelo: 13->32->16->8->3 (softmax)")
    print(f" Clases: {', '.join(CLASS_NAMES)}")
    print(f" HTTP: http://0.0.0.0:5000/deploy-model")
    print(f" MQTT: {MQTT_BROKER}:{MQTT_PORT}")
    print(f" PC: {url_coordinador}")
    print("=" * 60)

    mqtt_client = mqtt.Client()
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message

    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except ConnectionRefusedError:
        print(f"ERROR: Mosquitto no disponible en {MQTT_BROKER}:{MQTT_PORT}")
        exit(1)

    mqtt_client.loop_start()

    server = HTTPServer(("0.0.0.0", 5000), DeployModelHandler)
    print("\n[HTTP] Servidor listo en puerto 5000")
    print("[MQTT] Esperando deltas de ESP32s...\n")
    server.serve_forever()
