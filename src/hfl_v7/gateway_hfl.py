"""
=============================================================================
 gateway_hfl.py — Raspberry Pi 4 (Edge Gateway) — HFL v7 (3 Clases)
=============================================================================
 SIN ASCON — Rama para medición de tiempos sin cifrado
 Topología: 2x Nodos ESP32 -> Raspberry Pi 4 (Aquí) -> PC
 Modelo: 13 -> 32 -> 16 -> 8 -> 3 (softmax)
 Capas FL: W3(16,8) + b3(8) + W4(8,3) + b4(3)
 
 Ejecutar: python gateway_hfl.py
=============================================================================
"""
import numpy as np
import requests
import json
import threading
import paho.mqtt.client as mqtt
from http.server import HTTPServer, BaseHTTPRequestHandler
import tensorflow as tf
import time

# ====================== CONFIGURACIÓN ======================
GATEWAY_ID = "gateway_A"
IP_PC = "192.168.40.95"
PORT_PC = "8001"
url_servidor = f"http://{IP_PC}:{PORT_PC}/aggregate-from-gateway"

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
TOPIC_FEATURES = "fl/features"
TOPIC_GLOBAL_MODEL = "fl/global_model"

# ====================== DATASET LOCAL ======================
FEATURE_COUNT = 13
CLASS_NAMES = ["normal", "mqtt_bruteforce", "scan_A"]
MIN_PKTS_FOR_ML = 1
RULE_PKTS_ALERT = 100

X_train_buffer = []
Y_train_buffer = []
SAMPLES_PER_UPDATE = 40

current_round = 0

# ====================== TRACKING DE NODOS ======================
node_stats = {}

def update_node_stats(client_id, label):
    if client_id not in node_stats:
        node_stats[client_id] = {"samples": 0, "last_seen": 0, "labels": {0:0, 1:0, 2:0}}
    node_stats[client_id]["samples"] += 1
    node_stats[client_id]["last_seen"] = time.time()
    if label >= 0:
        node_stats[client_id]["labels"][label] = node_stats[client_id]["labels"].get(label, 0) + 1

def print_node_summary():
    now = time.time()
    print(f"\n{'─'*60}")
    print(f" RESUMEN DE NODOS CONECTADOS ({len(node_stats)} activos)")
    print(f"{'─'*60}")
    for nid, info in node_stats.items():
        age = now - info["last_seen"]
        status = "●" if age < 15 else "○"
        dist = ", ".join(f"{CLASS_NAMES[k]}:{v}" for k,v in info["labels"].items() if v > 0)
        print(f"  {status} {nid:30s} | total={info['samples']:4d} | {dist}")
    print(f"{'─'*60}")

# ====================== MODELO KERAS =======================
print("[GATEWAY] Cargando modelo base ids_3class.keras...")
try:
    model = tf.keras.models.load_model("ids_3class.keras")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    print("[GATEWAY] Modelo cargado exitosamente.")
except Exception as e:
    print(f"[ERROR] No se pudo cargar ids_3class.keras: {e}")
    print("[GATEWAY] Creando modelo desde cero...")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(FEATURE_COUNT,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

mqtt_client = None

# ====================== HTTP SERVER (Recibe global del PC) ======================
class DeployModelHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        global current_round
        if self.path == "/deploy-model":
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            
            t0 = time.perf_counter()
            data = json.loads(body.decode('utf-8'))
            parse_ms = (time.perf_counter() - t0) * 1000
            
            W3 = np.array(data["W3"], dtype=np.float32)
            b3 = np.array(data["b3"], dtype=np.float32)
            W4 = np.array(data["W4"], dtype=np.float32)
            b4 = np.array(data["b4"], dtype=np.float32)
            current_round = data.get("round", current_round + 1)

            print(f"\n{'='*60}")
            print(f" MODELO GLOBAL RECIBIDO DEL PC (Ronda {current_round})")
            print(f" Recepcion: {parse_ms:.3f}ms | {len(body)}B (sin cifrado)")
            
            dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
            if len(dense_layers) >= 2:
                dense_layers[-2].set_weights([W3, b3])
                dense_layers[-1].set_weights([W4, b4])
            
            print(f" Pesos Keras actualizados. W4 mag: {np.mean(np.abs(W4)):.6f}")
            print(f"{'='*60}")

            broadcast_model_to_esp32s(W3, b3, W4, b4)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "round": current_round}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args): pass


def broadcast_model_to_esp32s(W3, b3, W4, b4):
    payload = {
        "round": current_round,
        "W3": W3.tolist(), "b3": b3.tolist(),
        "W4": W4.tolist(), "b4": b4.tolist()
    }
    payload_str = json.dumps(payload)
    
    t0 = time.perf_counter()
    if mqtt_client and mqtt_client.is_connected():
        mqtt_client.publish(TOPIC_GLOBAL_MODEL, payload_str, qos=1)
        pub_ms = (time.perf_counter() - t0) * 1000
        print(f"[MQTT] Modelo global publicado para ESP32s ({len(payload_str)}B, {pub_ms:.3f}ms, sin cifrado)")


# ====================== HEURÍSTICA Y ENTRENAMIENTO ======================
def heuristicLabel(features):
    pkts = int(features[0])
    meanPktLen = features[5]
    numPsh = features[7]

    if pkts >= 50 and numPsh >= 10:
        return 1  # mqtt_bruteforce
    if pkts <= 5 and meanPktLen <= 50 and numPsh <= 1:
        return 2  # scan_A
    if pkts <= 30 and meanPktLen >= 50:
        return 0  # normal
    return -1

def train_local_model():
    global X_train_buffer, Y_train_buffer

    X = np.array(X_train_buffer, dtype=np.float32)
    Y = np.array(Y_train_buffer, dtype=np.int32)
    
    X_train_buffer = []
    Y_train_buffer = []
    
    print(f"\n[ENTRENAMIENTO LOCAL] Iniciando fit sobre {len(X)} muestras...")
    
    hist = model.fit(X, Y, epochs=5, batch_size=8, verbose=1)
    final_acc = float(hist.history.get('accuracy', [0.0])[-1])
    final_loss = float(hist.history.get('loss', [0.0])[-1])
    
    W3, b3 = model.get_layer("dense_3").get_weights()
    W4, b4 = model.get_layer("dense_out").get_weights()
    
    print(f"[ENTRENAMIENTO LOCAL] Finalizado (Acc: {final_acc:.2%}, Loss: {final_loss:.4f})")
    
    payload = {
        "gateway_id": GATEWAY_ID,
        "num_samples": len(X),
        "round": current_round,
        "accuracy": final_acc,
        "loss": final_loss,
        "W3": W3.tolist(), "b3": b3.tolist(),
        "W4": W4.tolist(), "b4": b4.tolist()
    }
    
    t0 = time.perf_counter()
    payload_json = json.dumps(payload)
    serial_ms = (time.perf_counter() - t0) * 1000
    
    print(f"[SEND] Pesos serializados ({len(payload_json)}B, {serial_ms:.3f}ms). Enviando al servidor PC...")
    try:
        resp = requests.post(url_servidor, json=payload, timeout=5)
        print(f"-> Respuesta PC: {resp.json()}")
    except Exception as e:
        print(f"-> ERROR contactando PC: {e}")

# ====================== MQTT CALLBACKS ======================
def on_connect(client, userdata, flags, rc):
    print(f"\n[MQTT] Conectado a Mosquitto local (código: {rc})")
    client.subscribe(TOPIC_FEATURES)
    print(f"[MQTT] Suscrito a '{TOPIC_FEATURES}' para recibir de los ESP32\n")

def on_message(client, userdata, msg):
    try:
        t0 = time.perf_counter()
        data = json.loads(msg.payload.decode('utf-8'))
        parse_ms = (time.perf_counter() - t0) * 1000
        
        client_id = data.get("client_id", "unknown")
        features = data.get("features", [])
        
        if len(features) != FEATURE_COUNT: return
        
        label = heuristicLabel(features)
        update_node_stats(client_id, label)
        
        if label >= 0 and features[0] >= MIN_PKTS_FOR_ML:
            X_train_buffer.append(features)
            Y_train_buffer.append(label)
            
            print(f"[DATASET] {client_id} -> {CLASS_NAMES[label]} | Buffer: {len(X_train_buffer)}/{SAMPLES_PER_UPDATE} | {parse_ms:.3f}ms (sin cifrado)")
            
            if len(X_train_buffer) >= SAMPLES_PER_UPDATE:
                print_node_summary()
                train_local_model()
                
    except Exception as e:
        print(f"[ERROR] {e}")


# ====================== MAIN ======================
if __name__ == "__main__":
    print("=" * 60)
    print(f" GATEWAY HFL v7 [{GATEWAY_ID}] - RASPBERRY PI 4")
    print(f" *** SIN CIFRADO ASCON — Medición de baseline ***")
    print(f" Nodos esperados: normal_1, simulator_1 (2 nodos)")
    print(f" Buffer: {SAMPLES_PER_UPDATE} muestras")
    print(f" PC: {url_servidor}")
    print("=" * 60)

    mqtt_client = mqtt.Client()
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message

    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
    except ConnectionRefusedError:
        print(f"ERROR: Mosquitto no disponible en {MQTT_BROKER}:{MQTT_PORT}.")

    server = HTTPServer(("0.0.0.0", 5000), DeployModelHandler)
    print("\n[HTTP] Servidor listo en puerto 5000 para recibir de PC")
    print("[MQTT] Esperando características de Nodos ESP32...\n")
    server.serve_forever()
