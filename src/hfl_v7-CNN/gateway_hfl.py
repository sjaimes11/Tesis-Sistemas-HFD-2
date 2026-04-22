"""
=============================================================================
 gateway_hfl.py — Raspberry Pi 4 (Edge Gateway) — HFL v7-CNN (3 Clases)
=============================================================================
 Modelo: CNN-1D  (13,1) -> Conv32 -> BN -> Conv16 -> BN -> GAP -> Dense8 -> Dense3
 Capas FL: dense_1 (GAP->8) + dense_out (8->3)   ← equivalente a W3/W4 del MLP
 Seguridad: ASCON-128 en todos los canales (ESP32<->RPi, RPi<->PC)

 DIFERENCIA clave vs hfl_v7-RN (MLP):
   - El input se reshapea a (N, 13, 1) antes de model.fit()
   - Las capas federadas se llaman "dense_1" y "dense_out" en vez de
     "dense_3" y "dense_out"
   - El servidor (server_hfl.py) sigue siendo idéntico al de hfl_v7-RN
     porque solo intercambia W_dense1 / W_dense_out (mismas dimensiones: 16->8->3)

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
from pathlib import Path
from ascon128 import encrypt as ascon_encrypt, decrypt as ascon_decrypt, generate_nonce
from ascon_metrics import AsconMetrics
from model_results_logger import ModelResultsLogger

# ====================== CONFIGURACIÓN ======================
GATEWAY_ID = "gateway_A"
IP_PC = "192.168.40.95"
PORT_PC = "8001"
url_servidor = f"http://{IP_PC}:{PORT_PC}/aggregate-from-gateway"

metrics = AsconMetrics("gateway", suffix=GATEWAY_ID)
model_results = ModelResultsLogger("gateway", suffix=GATEWAY_ID)

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

# ====================== MODELO KERAS (CNN-1D) =======================
print("[GATEWAY-CNN] Cargando modelo base CNN-1D...")
try:
    model = load_base_cnn_model()
    print("[GATEWAY-CNN] Modelo CNN-1D cargado exitosamente.")
except Exception as e:
    print(f"[ERROR] No se pudo cargar ningun artefacto del modelo CNN: {e}")
    print("[GATEWAY-CNN] Creando modelo CNN-1D desde cero...")
    model = build_cnn_model()

mqtt_client = None


def build_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(FEATURE_COUNT, 1)),
        tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding='same', name='conv_1'),
        tf.keras.layers.BatchNormalization(name='bn_conv_1'),
        tf.keras.layers.Conv1D(16, kernel_size=3, activation='relu', padding='same', name='conv_2'),
        tf.keras.layers.BatchNormalization(name='bn_conv_2'),
        tf.keras.layers.GlobalAveragePooling1D(name='gap'),
        tf.keras.layers.Dense(8, activation='relu', name='dense_1'),
        tf.keras.layers.Dense(3, activation='softmax', name='dense_out'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def load_base_cnn_model():
    base_dir = Path(__file__).resolve().parent
    h5_path = base_dir / "ids_3class.h5"
    keras_path = base_dir / "ids_3class.keras"
    weights_path = base_dir / "ids_3class.weights.h5"

    if h5_path.exists():
        print(f"[GATEWAY-CNN] Intentando cargar fallback compatible: {h5_path.name}")
        model = tf.keras.models.load_model(str(h5_path), compile=False)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )
        return model

    if keras_path.exists():
        print(f"[GATEWAY-CNN] Intentando cargar modelo nativo Keras: {keras_path.name}")
        model = tf.keras.models.load_model(str(keras_path), compile=False)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )
        return model

    model = build_cnn_model()
    if weights_path.exists():
        print(f"[GATEWAY-CNN] Cargando pesos manualmente desde {weights_path.name}")
        model.load_weights(str(weights_path))
    return model

# ====================== HTTP SERVER (Recibe global cifrado del PC) ======================
class DeployModelHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        global current_round
        if self.path == "/deploy-model":
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            envelope = json.loads(body.decode('utf-8'))

            ct    = base64.b64decode(envelope["ct"])
            tag   = base64.b64decode(envelope["tag"])
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

            # ── CNN-1D: capas federadas son dense_1 y dense_out
            W_d1 = np.array(data["W_dense1"],   dtype=np.float32)
            b_d1 = np.array(data["b_dense1"],   dtype=np.float32)
            W_do = np.array(data["W_dense_out"], dtype=np.float32)
            b_do = np.array(data["b_dense_out"], dtype=np.float32)
            current_round = data.get("round", current_round + 1)

            print(f"\n{'='*60}")
            print(f" MODELO GLOBAL RECIBIDO DEL PC (Ronda {current_round})")
            print(f" ASCON: Descifrado y autenticado exitosamente")

            model.get_layer("dense_1").set_weights([W_d1, b_d1])
            model.get_layer("dense_out").set_weights([W_do, b_do])

            print(f" Pesos Keras actualizados. W_dense_out mag: {np.mean(np.abs(W_do)):.6f}")
            print(f"{'='*60}")

            broadcast_model_to_esp32s(W_d1, b_d1, W_do, b_do)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "round": current_round}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args): pass


def broadcast_model_to_esp32s(W_d1, b_d1, W_do, b_do):
    """
    Publica el modelo global actualizado a los ESP32 via MQTT+ASCON.
    Nota: Los ESP32 usan solo las capas Dense (dense_1 y dense_out)
    para actualización parcial de pesos. Las capas Conv se mantienen fijas
    en el dispositivo Edge (vienen del modelo base compilado en C++).
    """
    global msg_counter
    payload = {
        "round": current_round,
        "W_dense1":   W_d1.tolist(), "b_dense1":   b_d1.tolist(),
        "W_dense_out": W_do.tolist(), "b_dense_out": b_do.tolist(),
    }
    payload_bytes = json.dumps(payload).encode('utf-8')

    nonce = generate_nonce(int(time.time() * 1000), msg_counter)
    msg_counter += 1

    t0 = time.perf_counter()
    ciphertext, tag = ascon_encrypt(payload_bytes, ASCON_KEY, nonce)
    enc_ms = (time.perf_counter() - t0) * 1000

    envelope = {
        "ct":    base64.b64encode(ciphertext).decode('ascii'),
        "tag":   base64.b64encode(tag).decode('ascii'),
        "nonce": base64.b64encode(nonce).decode('ascii'),
    }
    envelope_str = json.dumps(envelope)

    metrics.record("RPi->ESP32", "encrypt", len(payload_bytes), len(envelope_str), enc_ms, current_round)

    if mqtt_client and mqtt_client.is_connected():
        mqtt_client.publish(TOPIC_GLOBAL_MODEL, envelope_str, qos=1)
        print(f"[MQTT] Modelo global cifrado (ASCON) publicado para ESP32s ({len(envelope_str)} bytes)")


# ====================== HEURÍSTICA Y ENTRENAMIENTO ======================
def heuristicLabel(features):
    pkts      = int(features[0])
    meanPktLen = features[5]
    numPsh    = features[7]

    if pkts >= 50 and numPsh >= 10:
        return 1  # mqtt_bruteforce
    if pkts <= 5 and meanPktLen <= 50 and numPsh <= 1:
        return 2  # scan_A
    if pkts <= 30 and meanPktLen >= 50:
        return 0  # normal
    return -1


def train_local_model():
    global X_train_buffer, Y_train_buffer, msg_counter

    X_raw = np.array(X_train_buffer, dtype=np.float32)
    Y     = np.array(Y_train_buffer, dtype=np.int32)

    X_train_buffer = []
    Y_train_buffer = []

    # ── CNN-1D: reshape a (N, 13, 1)
    X_cnn = X_raw.reshape(-1, FEATURE_COUNT, 1)

    print(f"\n[ENTRENAMIENTO LOCAL CNN] Iniciando fit sobre {len(X_cnn)} muestras...")

    hist = model.fit(X_cnn, Y, epochs=5, batch_size=8, verbose=1)
    final_acc  = float(hist.history.get('accuracy', [0.0])[-1])
    final_loss = float(hist.history.get('loss',     [0.0])[-1])

    W_d1, b_d1 = model.get_layer("dense_1").get_weights()
    W_do, b_do = model.get_layer("dense_out").get_weights()

    print(f"[ENTRENAMIENTO LOCAL CNN] Finalizado (Acc: {final_acc:.2%}, Loss: {final_loss:.4f})")
    model_results.record(
        stage="local_train",
        fl_round=current_round,
        num_samples=len(X_cnn),
        accuracy=final_acc,
        loss=final_loss,
        buffer_target=SAMPLES_PER_UPDATE,
    )

    # ── Payload FL: misma estructura que MLP pero con claves renombradas
    payload = {
        "gateway_id":  GATEWAY_ID,
        "num_samples": len(X_cnn),
        "round":       current_round,
        "accuracy":    final_acc,
        "loss":        final_loss,
        "W_dense1":    W_d1.tolist(), "b_dense1":    b_d1.tolist(),
        "W_dense_out": W_do.tolist(), "b_dense_out": b_do.tolist(),
    }
    payload_bytes = json.dumps(payload).encode('utf-8')

    nonce = generate_nonce(int(time.time() * 1000), msg_counter)
    msg_counter += 1

    t0 = time.perf_counter()
    ciphertext, tag = ascon_encrypt(payload_bytes, ASCON_KEY, nonce)
    enc_ms = (time.perf_counter() - t0) * 1000

    envelope = {
        "ct":    base64.b64encode(ciphertext).decode('ascii'),
        "tag":   base64.b64encode(tag).decode('ascii'),
        "nonce": base64.b64encode(nonce).decode('ascii'),
    }
    envelope_json = json.dumps(envelope).encode('utf-8')
    metrics.record("RPi->PC", "encrypt", len(payload_bytes), len(envelope_json), enc_ms, current_round)

    print(f"[ASCON] Pesos CNN cifrados. Enviando al servidor PC...")
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

        ct    = base64.b64decode(envelope["ct"])
        tag   = base64.b64decode(envelope["tag"])
        nonce = base64.b64decode(envelope["nonce"])

        t0 = time.perf_counter()
        plaintext = ascon_decrypt(ct, ASCON_KEY, nonce, tag)
        dec_ms = (time.perf_counter() - t0) * 1000

        if plaintext is None:
            print("[ERROR] ASCON: Tag inválido desde ESP32. Mensaje rechazado.")
            return

        data      = json.loads(plaintext.decode('utf-8'))
        client_id = data.get("client_id", "unknown")
        features  = data.get("features", [])

        if len(features) != FEATURE_COUNT: return

        label      = heuristicLabel(features)
        label_name = CLASS_NAMES[label] if 0 <= label < len(CLASS_NAMES) else "unknown"
        metrics.record("ESP32->RPi", "decrypt", len(plaintext), len(msg.payload),
                       dec_ms, current_round, client_id=client_id,
                       sample_label=label, sample_label_name=label_name)
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
    print(f" GATEWAY HFL-CNN v7 [{GATEWAY_ID}] - RASPBERRY PI 4")
    print(f" Modelo: CNN-1D  (13,1)->Conv32->Conv16->GAP->Dense8->Dense3")
    print(f" Capas FL: dense_1 + dense_out")
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
