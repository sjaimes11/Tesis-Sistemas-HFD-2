"""
=============================================================================
 aggregator_hfl.py — Raspberry Pi (Fog Aggregator) — HFL Bidireccional
=============================================================================
 Rol: 
   1. Recibe modelo global del PC via HTTP POST /deploy-model
   2. Publica el modelo a los ESP32s via MQTT (fl/global_model)
   3. Escucha deltas de ESP32s via MQTT (fl/updates)
   4. Hace FedAvg y envía resultado al PC via HTTP POST
 
 Ejecutar: python3 aggregator_hfl.py
=============================================================================
 Requiere: pip3 install paho-mqtt numpy requests
 Mosquitto debe estar corriendo: sudo systemctl start mosquitto
 (NO necesita Flask — usa http.server de la stdlib)
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
OUTPUT_UNITS = 1

# ====================== MODELO GLOBAL (recibido del PC) ======================
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
# IP del PC Coordinador (cambiar a la IP actual de tu PC)
IP_PC = "192.168.1.22"
PORT_PC = "8001"
url_coordinador = f"http://{IP_PC}:{PORT_PC}/aggregate-from-pi"

# MQTT local (Mosquitto en la Raspberry Pi)
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
TOPIC_UPDATES = "fl/updates"
TOPIC_GLOBAL_MODEL = "fl/global_model"

# Cliente MQTT global
mqtt_client = None

# ====================== HTTP SERVER (stdlib, no Flask) ======================

class DeployModelHandler(BaseHTTPRequestHandler):
    """Maneja POST /deploy-model del PC Coordinador."""

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

            # Reset acumuladores
            W3_update_sum.fill(0); b3_update_sum.fill(0)
            W4_update_sum.fill(0); b4_update_sum.fill(0)
            total_samples_this_round = 0; updates_received = 0

            print(f"\n{'='*60}")
            print(f" MODELO GLOBAL RECIBIDO DEL PC (Ronda {current_round})")
            print(f" W3 mag: {np.mean(np.abs(W3_global)):.6f}")
            print(f" W4 mag: {np.mean(np.abs(W4_global)):.6f}")
            print(f"{'='*60}")

            broadcast_model_to_esp32s()

            resp = json.dumps({"status": "ok", "round": current_round, "message": "Modelo publicado a ESP32s"})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(resp.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Silenciar logs de acceso HTTP
        pass


def broadcast_model_to_esp32s():
    """Publica el modelo global en fl/global_model para que los ESP32 lo reciban."""
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
        print(f"[MQTT] Modelo publicado en '{TOPIC_GLOBAL_MODEL}' ({len(payload_str)} bytes)")
        print(f"       Esperando deltas de ESP32s...\n")
    else:
        print("[ERROR] MQTT no conectado")


# ====================== MQTT CALLBACKS ======================

def on_connect(client, userdata, flags, rc):
    print(f"\n[MQTT] Conectado a Mosquitto (codigo: {rc})")
    client.subscribe(TOPIC_UPDATES)
    print(f"[MQTT] Suscrito a '{TOPIC_UPDATES}'\n")


def on_message(client, userdata, msg):
    """Recibe deltas de los ESP32 Brokers."""
    global W3_update_sum, b3_update_sum, W4_update_sum, b4_update_sum
    global total_samples_this_round, updates_received

    try:
        data = json.loads(msg.payload.decode('utf-8'))

        client_id = data.get("client_id", "unknown")
        num_samples = data.get("num_samples", 0)
        round_num = data.get("round", 0)

        print(f"[MQTT] Delta de '{client_id}' | Ronda: {round_num} | Muestras: {num_samples}")

        weight_delta = data.get("weight_delta", {})
        delta_W3 = np.array(weight_delta.get("W3", []), dtype=np.float32)
        delta_b3 = np.array(weight_delta.get("b3", []), dtype=np.float32)
        delta_W4 = np.array(weight_delta.get("W4", []), dtype=np.float32)
        delta_b4 = np.array(weight_delta.get("b4", []), dtype=np.float32)

        if delta_W4.ndim == 1 and delta_W4.shape[0] == L3_UNITS:
            delta_W4 = delta_W4.reshape(L3_UNITS, OUTPUT_UNITS)

        ok = True
        if delta_W3.shape != (L2_UNITS, L3_UNITS):
            print(f"  ERROR W3: {delta_W3.shape}"); ok = False
        if delta_b3.shape != (L3_UNITS,):
            print(f"  ERROR b3: {delta_b3.shape}"); ok = False
        if delta_W4.shape != (L3_UNITS, OUTPUT_UNITS):
            print(f"  ERROR W4: {delta_W4.shape}"); ok = False
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

        print(f"  Acumulado: {updates_received}/{MIN_UPDATES_PER_ROUND} updates")

        if updates_received >= MIN_UPDATES_PER_ROUND:
            do_fedavg_and_send()

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback; traceback.print_exc()


def do_fedavg_and_send():
    """Ejecuta FedAvg y envía el modelo agregado al PC Coordinador."""
    global W3_global, b3_global, W4_global, b4_global
    global W3_update_sum, b3_update_sum, W4_update_sum, b4_update_sum
    global total_samples_this_round, updates_received

    print(f"\n{'='*60}")
    print(f" FEDAVG - {updates_received} updates, {total_samples_this_round} muestras")
    print(f"{'='*60}")

    W3_global += W3_update_sum / total_samples_this_round
    b3_global += b3_update_sum / total_samples_this_round
    W4_global += W4_update_sum / total_samples_this_round
    b4_global += b4_update_sum / total_samples_this_round

    W3_update_sum.fill(0); b3_update_sum.fill(0)
    W4_update_sum.fill(0); b4_update_sum.fill(0)
    total_samples_this_round = 0; updates_received = 0

    print(f"  W3 mag: {np.mean(np.abs(W3_global)):.6f}")
    print(f"  W4 mag: {np.mean(np.abs(W4_global)):.6f}")

    payload = {
        "round": current_round,
        "model": "binary_13_32_16_8_1",
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
    print(" AGREGADOR HFL - Raspberry Pi (Fog)")
    print(f" HTTP Server: http://0.0.0.0:5000/deploy-model")
    print(f" MQTT: {MQTT_BROKER}:{MQTT_PORT}")
    print(f" PC Coordinador: {url_coordinador}")
    print("=" * 60)

    # 1. MQTT client
    mqtt_client = mqtt.Client()
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message

    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except ConnectionRefusedError:
        print(f"ERROR: Mosquitto no disponible en {MQTT_BROKER}:{MQTT_PORT}")
        print("  sudo systemctl start mosquitto")
        exit(1)

    mqtt_client.loop_start()

    # 2. HTTP server (stdlib, sin Flask)
    server = HTTPServer(("0.0.0.0", 5000), DeployModelHandler)
    print("\n[HTTP] Servidor listo en puerto 5000")
    print("[MQTT] Esperando deltas de ESP32s...\n")
    server.serve_forever()

