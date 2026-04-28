"""
=============================================================================
 gateway_hfl.py — Raspberry Pi 4 (Edge Gateway) — HFL v7-CNN (3 Clases)
=============================================================================
 Topología: 2x Nodos ESP32 -> Raspberry Pi 4 (Aquí) -> PC
 Nodos esperados: esp32_edge_normal_1, esp32_edge_simulator_1
 Modelo CNN-1D: 13 features -> Conv1D(32) -> Conv1D(16) -> GAP -> Dense(8) -> Dense(3)
 Capas FL: W_dense1(16,8) + b_dense1(8) + W_dense_out(8,3) + b_dense_out(3)
 Modo baseline sin ASCON: JSON plano en todos los canales (ESP32<->RPi, RPi<->PC)
 
 Ejecutar: python gateway_hfl.py
=============================================================================
 Requiere: pip install paho-mqtt numpy requests tensorflow
=============================================================================
"""
import numpy as np
import requests
import json
import paho.mqtt.client as mqtt
from http.server import HTTPServer, BaseHTTPRequestHandler
import tensorflow as tf
import time
from pathlib import Path
from plain_metrics import PlainMetrics
from model_results_logger import ModelResultsLogger

# ====================== CONFIGURACIÓN ======================
GATEWAY_ID = "gateway_A"
IP_PC = "192.168.40.95"
PORT_PC = "8002"
PLAIN_AGGREGATE_PATH = "/aggregate-from-gateway-plain"
PLAIN_DEPLOY_PATH = "/deploy-model-plain"
url_servidor = f"http://{IP_PC}:{PORT_PC}{PLAIN_AGGREGATE_PATH}"

metrics = PlainMetrics("gateway", suffix=GATEWAY_ID)
model_results = ModelResultsLogger("gateway", suffix=GATEWAY_ID)

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
TOPIC_FEATURES = "fl/features_plain"
TOPIC_GLOBAL_MODEL = "fl/global_model_plain"

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

# ====================== MODELO KERAS CNN-1D =======================
def build_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(FEATURE_COUNT, 1), name='input_features'),
        tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding='same', name='conv_1'),
        tf.keras.layers.BatchNormalization(name='bn_conv_1'),
        tf.keras.layers.Dropout(0.3, name='drop_1'),
        tf.keras.layers.Conv1D(16, kernel_size=3, activation='relu', padding='same', name='conv_2'),
        tf.keras.layers.BatchNormalization(name='bn_conv_2'),
        tf.keras.layers.Dropout(0.2, name='drop_2'),
        tf.keras.layers.GlobalAveragePooling1D(name='gap'),
        tf.keras.layers.Dense(8, activation='relu', name='dense_1'),
        tf.keras.layers.Dense(3, activation='softmax', name='dense_out'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def load_base_cnn_model():
    base_dir = Path(__file__).resolve().parent
    weights_path = base_dir / "ids_3class_cnn.weights.h5"
    keras_path = base_dir / "ids_3class.keras"

    if keras_path.exists():
        print(f"[GATEWAY-CNN] Intentando cargar modelo nativo Keras: {keras_path.name}")
        try:
            loaded_model = tf.keras.models.load_model(str(keras_path), compile=False)
            loaded_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                          loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            return loaded_model
        except Exception as e:
            print(f"[GATEWAY-CNN] No se pudo cargar .keras directamente: {e}")

    model = build_cnn_model()
    if weights_path.exists():
        print(f"[GATEWAY-CNN] Cargando pesos manualmente desde {weights_path.name}")
        model.load_weights(str(weights_path))
    return model


print("[GATEWAY-CNN] Cargando modelo base CNN-1D...")
try:
    model = load_base_cnn_model()
    print("[GATEWAY-CNN] Modelo CNN-1D cargado exitosamente.")
except Exception as e:
    print(f"[ERROR] No se pudo cargar modelo CNN-1D: {e}")
    print("[GATEWAY-CNN] Creando modelo CNN-1D desde cero...")
    model = build_cnn_model()

mqtt_client = None

# ====================== HTTP SERVER (Recibe global plano del PC) ======================
class DeployModelHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        global current_round
        if self.path == PLAIN_DEPLOY_PATH:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            t0 = time.perf_counter()
            data = json.loads(body.decode('utf-8'))
            parse_ms = (time.perf_counter() - t0) * 1000
            metrics.record("PC->RPi", "deserialize", len(body), parse_ms, current_round)
            Wd1 = np.array(data["W_dense1"],    dtype=np.float32)
            bd1 = np.array(data["b_dense1"],    dtype=np.float32)
            Wdo = np.array(data["W_dense_out"], dtype=np.float32)
            bdo = np.array(data["b_dense_out"], dtype=np.float32)
            current_round = data.get("round", current_round + 1)

            print(f"\n{'='*60}")
            print(f" MODELO GLOBAL CNN-1D RECIBIDO DEL PC (Ronda {current_round})")
            print(" JSON plano recibido y parseado correctamente")
            
            dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
            if len(dense_layers) >= 2:
                dense_layers[-2].set_weights([Wd1, bd1])
                dense_layers[-1].set_weights([Wdo, bdo])
            
            print(f" Pesos Keras actualizados. Wdo mag: {np.mean(np.abs(Wdo)):.6f}")
            print(f"{'='*60}")

            broadcast_model_to_esp32s(Wd1, bd1, Wdo, bdo)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "round": current_round}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args): pass


def broadcast_model_to_esp32s(Wd1, bd1, Wdo, bdo):
    payload = {
        "round": current_round,
        "W_dense1": Wd1.tolist(), "b_dense1": bd1.tolist(),
        "W_dense_out": Wdo.tolist(), "b_dense_out": bdo.tolist()
    }
    t0 = time.perf_counter()
    payload_json = json.dumps(payload)
    serialize_ms = (time.perf_counter() - t0) * 1000
    metrics.record("RPi->ESP32", "serialize", len(payload_json.encode('utf-8')), serialize_ms, current_round)
    
    if mqtt_client and mqtt_client.is_connected():
        mqtt_client.publish(TOPIC_GLOBAL_MODEL, payload_json, qos=1)
        print(f"[MQTT] Modelo global CNN-1D plano publicado para ESP32s ({len(payload_json)} bytes)")


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
    global X_train_buffer, Y_train_buffer

    X = np.array(X_train_buffer, dtype=np.float32)
    Y = np.array(Y_train_buffer, dtype=np.int32)
    
    X_train_buffer = []
    Y_train_buffer = []

    # CNN-1D espera entrada con shape (N, 13, 1)
    X_cnn = X.reshape(-1, FEATURE_COUNT, 1)
    
    print(f"\n[ENTRENAMIENTO LOCAL CNN-1D] Iniciando fit sobre {len(X)} muestras...")
    
    hist = model.fit(X_cnn, Y, epochs=5, batch_size=8, verbose=1)
    final_acc = float(hist.history.get('accuracy', [0.0])[-1])
    final_loss = float(hist.history.get('loss', [0.0])[-1])
    
    dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
    Wd1, bd1 = dense_layers[-2].get_weights()
    Wdo, bdo = dense_layers[-1].get_weights()
    
    print(f"[ENTRENAMIENTO LOCAL CNN-1D] Finalizado (Acc: {final_acc:.2%}, Loss: {final_loss:.4f})")
    model_results.record(
        stage="local_train",
        fl_round=current_round,
        num_samples=len(X),
        accuracy=final_acc,
        loss=final_loss,
        buffer_target=SAMPLES_PER_UPDATE,
    )
    
    payload = {
        "gateway_id": GATEWAY_ID,
        "num_samples": len(X),
        "round": current_round,
        "accuracy": final_acc,
        "loss": final_loss,
        "W_dense1": Wd1.tolist(), "b_dense1": bd1.tolist(),
        "W_dense_out": Wdo.tolist(), "b_dense_out": bdo.tolist()
    }
    t0 = time.perf_counter()
    payload_json = json.dumps(payload)
    serialize_ms = (time.perf_counter() - t0) * 1000
    metrics.record("RPi->PC", "serialize", len(payload_json.encode('utf-8')), serialize_ms, current_round)
    
    print(f"[PLAIN] Pesos CNN-1D serializados. Enviando al servidor PC en {url_servidor} ...")
    try:
        resp = requests.post(url_servidor, json=payload, timeout=5)
        if resp.status_code == 422 and all(token in resp.text for token in ["ct", "tag", "nonce"]):
            print(
                "-> ERROR: el servidor remoto sigue esperando ASCON. "
                f"Ejecuta hfl_v7-no-ascon-CNN/server_hfl.py en el puerto {PORT_PC}."
            )
            return
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
        label_name = CLASS_NAMES[label] if 0 <= label < len(CLASS_NAMES) else "unknown"
        metrics.record(
            "ESP32->RPi",
            "deserialize",
            len(msg.payload),
            parse_ms,
            current_round,
            client_id=client_id,
            sample_label=label,
            sample_label_name=label_name,
        )
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
    print(f" GATEWAY HFL v7-CNN [{GATEWAY_ID}] - RASPBERRY PI 4")
    print(f" Nodos esperados: normal_1, simulator_1 (2 nodos)")
    print(f" Buffer: {SAMPLES_PER_UPDATE} muestras | Modelo: CNN-1D")
    print(" Modo baseline sin ASCON: JSON plano (ESP32<->RPi, RPi<->PC)")
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
    print(f"\n[HTTP] Servidor listo en puerto 5000 para recibir de PC en {PLAIN_DEPLOY_PATH}")
    print("[MQTT] Esperando características de Nodos ESP32...\n")
    server.serve_forever()
