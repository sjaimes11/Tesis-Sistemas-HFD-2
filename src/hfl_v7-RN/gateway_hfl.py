"""
=============================================================================
 gateway_hfl.py — Raspberry Pi 4 (Edge Gateway) — HFL v7 (3 Clases)
=============================================================================
 Topología: 2x Nodos ESP32 -> Raspberry Pi 4 (Aquí) -> PC
 Nodos esperados: esp32_edge_normal_1, esp32_edge_simulator_1
 Modelo: 13 -> 32 -> 16 -> 8 -> 3 (softmax)
 Capas FL: W3(16,8) + b3(8) + W4(8,3) + b4(3)
 Seguridad: ASCON-128 en todos los canales (ESP32<->RPi, RPi<->PC)
 
 Ejecutar: python gateway_hfl.py
=============================================================================
 Requiere: pip install paho-mqtt numpy requests tensorflow
=============================================================================
"""
import numpy as np
import requests
import json
import threading
import paho.mqtt.client as mqtt
from http.server import HTTPServer, BaseHTTPRequestHandler
import tensorflow as tf
import base64
import time
from ascon128 import encrypt as ascon_encrypt, decrypt as ascon_decrypt, generate_nonce
from ascon_metrics import AsconMetrics

metrics = AsconMetrics("gateway")

# ====================== CONFIGURACIÓN ======================
GATEWAY_ID = "gateway_A"
IP_PC = "192.168.40.95"
PORT_PC = "8001"
url_servidor = f"http://{IP_PC}:{PORT_PC}/aggregate-from-gateway"

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
TOPIC_FEATURES = "fl/features"
TOPIC_GLOBAL_MODEL = "fl/global_model"

# Clave ASCON pre-compartida (misma que en ESP32 y PC)
ASCON_KEY = bytes([0xA1, 0xB2, 0xC3, 0xD4, 0xE5, 0xF6, 0x07, 0x18,
                   0x29, 0x3A, 0x4B, 0x5C, 0x6D, 0x7E, 0x8F, 0x90])

msg_counter = 0

# ====================== DATASET LOCAL ======================
FEATURE_COUNT = 13
CLASS_NAMES = ["normal", "mqtt_bruteforce", "scan_A"]
MIN_PKTS_FOR_ML = 1
RULE_PKTS_ALERT = 100

X_train_buffer = []
Y_train_buffer = []
SAMPLES_PER_UPDATE = 40  # Con 2 nodos enviando c/5s, buffer se llena en ~1.5 min

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
        tf.keras.layers.Dense(8, activation='relu'),  # W3, b3 (index 2)
        tf.keras.layers.Dense(3, activation='softmax') # W4, b4 (index 3)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

mqtt_client = None

# ====================== HTTP SERVER (Recibe global cifrado del PC) ======================
class DeployModelHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        global current_round
        if self.path == "/deploy-model":
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            envelope = json.loads(body.decode('utf-8'))
            
            ct = base64.b64decode(envelope["ct"])
            tag = base64.b64decode(envelope["tag"])
            nonce = base64.b64decode(envelope["nonce"])
            
            t0 = time.perf_counter()
            plaintext = ascon_decrypt(ct, ASCON_KEY, nonce, tag)
            dec_ms = (time.perf_counter() - t0) * 1000
            
            if plaintext is None:
                print("[ERROR] ASCON: Tag inválido desde PC. Mensaje rechazado.")
                self.send_response(403)
                self.end_headers()
                return
            
            metrics.record("PC->RPi", "decrypt", len(plaintext), len(body), dec_ms, current_round)
            data = json.loads(plaintext.decode('utf-8'))
            W3 = np.array(data["W3"], dtype=np.float32)
            b3 = np.array(data["b3"], dtype=np.float32)
            W4 = np.array(data["W4"], dtype=np.float32)
            b4 = np.array(data["b4"], dtype=np.float32)
            current_round = data.get("round", current_round + 1)

            print(f"\n{'='*60}")
            print(f" MODELO GLOBAL RECIBIDO DEL PC (Ronda {current_round})")
            print(f" ASCON: Descifrado y autenticado exitosamente")
            
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
    global msg_counter
    payload = {
        "round": current_round,
        "W3": W3.tolist(), "b3": b3.tolist(),
        "W4": W4.tolist(), "b4": b4.tolist()
    }
    payload_bytes = json.dumps(payload).encode('utf-8')
    
    nonce = generate_nonce(int(time.time() * 1000), msg_counter)
    msg_counter += 1
    
    t0 = time.perf_counter()
    ciphertext, tag = ascon_encrypt(payload_bytes, ASCON_KEY, nonce)
    enc_ms = (time.perf_counter() - t0) * 1000
    
    envelope = {
        "ct": base64.b64encode(ciphertext).decode('ascii'),
        "tag": base64.b64encode(tag).decode('ascii'),
        "nonce": base64.b64encode(nonce).decode('ascii')
    }
    envelope_str = json.dumps(envelope)
    
    metrics.record("RPi->ESP32", "encrypt", len(payload_bytes), len(envelope_str), enc_ms, current_round)
    
    if mqtt_client and mqtt_client.is_connected():
        mqtt_client.publish(TOPIC_GLOBAL_MODEL, envelope_str, qos=1)
        print(f"[MQTT] Modelo global cifrado (ASCON) publicado para ESP32s ({len(envelope_str)} bytes)")


# ====================== HEURÍSTICA Y ENTRENAMIENTO ======================
# Basada en estadísticas reales de los datasets:
#   Normal:     pkts~5,   IAT~0.0004s, pktLen~63, PSH~2,  RST=0
#   Bruteforce: pkts~345, IAT~3.38s,   pktLen~60, PSH~69, RST=0
#   Scan_A:     pkts~1,   IAT~0,       pktLen~44, PSH=0,  RST~0.4
def heuristicLabel(features):
    pkts = int(features[0])
    meanIat = features[1]
    meanPktLen = features[5]
    numPsh = features[7]

    # Bruteforce: muchos paquetes + muchos PSH (MQTT login attempts)
    if pkts >= 50 and numPsh >= 10:
        return 1  # mqtt_bruteforce

    # Scan_A: pocos paquetes + paquetes pequeños + sin PSH
    if pkts <= 5 and meanPktLen <= 50 and numPsh <= 1:
        return 2  # scan_A

    # Normal: pocos paquetes + paquetes medianos + algo de PSH
    if pkts <= 30 and meanPktLen >= 50:
        return 0  # normal

    return -1

def train_local_model():
    global X_train_buffer, Y_train_buffer, msg_counter

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
    payload_bytes = json.dumps(payload).encode('utf-8')
    
    nonce = generate_nonce(int(time.time() * 1000), msg_counter)
    msg_counter += 1
    
    t0 = time.perf_counter()
    ciphertext, tag = ascon_encrypt(payload_bytes, ASCON_KEY, nonce)
    enc_ms = (time.perf_counter() - t0) * 1000
    
    envelope = {
        "ct": base64.b64encode(ciphertext).decode('ascii'),
        "tag": base64.b64encode(tag).decode('ascii'),
        "nonce": base64.b64encode(nonce).decode('ascii')
    }
    envelope_json = json.dumps(envelope).encode('utf-8')
    metrics.record("RPi->PC", "encrypt", len(payload_bytes), len(envelope_json), enc_ms, current_round)
    
    print(f"[ASCON] Pesos cifrados. Enviando al servidor PC...")
    try:
        resp = requests.post(url_servidor, json=envelope, timeout=5)
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
        envelope = json.loads(msg.payload.decode('utf-8'))
        
        ct = base64.b64decode(envelope["ct"])
        tag = base64.b64decode(envelope["tag"])
        nonce = base64.b64decode(envelope["nonce"])
        
        t0 = time.perf_counter()
        plaintext = ascon_decrypt(ct, ASCON_KEY, nonce, tag)
        dec_ms = (time.perf_counter() - t0) * 1000
        
        if plaintext is None:
            print("[ERROR] ASCON: Tag inválido desde ESP32. Mensaje rechazado.")
            return
        
        metrics.record("ESP32->RPi", "decrypt", len(plaintext), len(msg.payload), dec_ms, current_round)
        data = json.loads(plaintext.decode('utf-8'))
        client_id = data.get("client_id", "unknown")
        features = data.get("features", [])
        
        if len(features) != FEATURE_COUNT: return
        
        label = heuristicLabel(features)
        update_node_stats(client_id, label)
        
        if label >= 0 and features[0] >= MIN_PKTS_FOR_ML:
            X_train_buffer.append(features)
            Y_train_buffer.append(label)
            
            print(f"[DATASET] {client_id} -> {CLASS_NAMES[label]} | Buffer: {len(X_train_buffer)}/{SAMPLES_PER_UPDATE} | Nodos: {len(node_stats)}")
            
            if len(X_train_buffer) >= SAMPLES_PER_UPDATE:
                print_node_summary()
                train_local_model()
                metrics.print_live_summary()
                
    except Exception as e:
        print(f"[ERROR] {e}")


# ====================== MAIN ======================
if __name__ == "__main__":
    print("=" * 60)
    print(f" GATEWAY HFL v7 [{GATEWAY_ID}] - RASPBERRY PI 4")
    print(f" Nodos esperados: normal_1, simulator_1 (2 nodos)")
    print(f" Buffer: {SAMPLES_PER_UPDATE} muestras")
    print(f" Seguridad: ASCON-128 (ESP32<->RPi, RPi<->PC)")
    print(f" PC: {url_servidor}")
    print("=" * 60)

    mqtt_client = mqtt.Client()
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message

    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
    except ConnectionRefusedError:
        print(f"ERROR: Mosquitto no disponible en {MQTT_BROKER}:{MQTT_PORT}. Por favor inicia 'sudo systemctl start mosquitto'")

    server = HTTPServer(("0.0.0.0", 5000), DeployModelHandler)
    print("\n[HTTP] Servidor listo en puerto 5000 para recibir de PC")
    print("[MQTT] Esperando características de Nodos ESP32...\n")
    server.serve_forever()
