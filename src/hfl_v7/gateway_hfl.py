"""
=============================================================================
 gateway_hfl.py — Raspberry Pi 4 (Edge Gateway) — HFL v7 (3 Clases)
=============================================================================
 Topología: Nodos ESP32 -> Raspberry Pi 4 (Aqúi) -> PC
 Modelo: 13 -> 32 -> 16 -> 8 -> 3 (softmax)
 Capas FL: W3(16,8) + b3(8) + W4(8,3) + b4(3)
 
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

# ====================== CONFIGURACIÓN ======================
GATEWAY_ID = "gateway_A"  # Cambiar a "gateway_B" en la otra Raspberry
IP_PC = "192.168.40.95"     # <- Cambiar por IP del PC (server_hfl.py)
PORT_PC = "8001"
url_servidor = f"http://{IP_PC}:{PORT_PC}/aggregate-from-gateway"

MQTT_BROKER = "localhost"  # Mosquitto corriendo en esta RPi 4
MQTT_PORT = 1883
TOPIC_FEATURES = "fl/features"
TOPIC_GLOBAL_MODEL = "fl/global_model"

# ====================== DATASET LOCAL ======================
FEATURE_COUNT = 13
CLASS_NAMES = ["normal", "mqtt_bruteforce", "scan_A"]
MIN_PKTS_FOR_ML = 10
RULE_PKTS_ALERT = 200

# Buffers de entrenamiento local
X_train_buffer = []
Y_train_buffer = []
SAMPLES_PER_UPDATE = 50  # Entrenar y enviar pesos cada 50 muestras

current_round = 0

# ====================== MODELO KERAS =======================
print("[GATEWAY] Cargando modelo base ids_3class.keras...")
try:
    model = tf.keras.models.load_model("ids_3class.keras")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
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
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

mqtt_client = None

# ====================== HTTP SERVER (Recibe global) ======================
class DeployModelHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        global current_round
        if self.path == "/deploy-model":
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            W3 = np.array(data["W3"], dtype=np.float32)
            b3 = np.array(data["b3"], dtype=np.float32)
            W4 = np.array(data["W4"], dtype=np.float32)
            b4 = np.array(data["b4"], dtype=np.float32)
            current_round = data.get("round", current_round + 1)

            print(f"\n{'='*60}")
            print(f" MODELO GLOBAL RECIBIDO DEL PC (Ronda {current_round})")
            
            # Buscamos de forma segura solo las capas Dense (ignorando Dropouts/Activations sueltas)
            dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
            if len(dense_layers) >= 2:
                dense_layers[-2].set_weights([W3, b3])
                dense_layers[-1].set_weights([W4, b4])
            
            print(f" Pesos Keras actualizados. W4 mag: {np.mean(np.abs(W4)):.6f}")
            print(f"{'='*60}")

            # Reenviar a los ESP32 para que actualicen su inferencia
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
    if mqtt_client and mqtt_client.is_connected():
        mqtt_client.publish(TOPIC_GLOBAL_MODEL, payload_str, qos=1)
        print(f"[MQTT] Modelo global publicado para ESP32s ({len(payload_str)} bytes)")


# ====================== HEURÍSTICA Y ENTRENAMIENTO ======================
def heuristicLabel(features):
    pkts = int(features[0])
    meanIat = features[1]
    numPsh = features[7]
    
    if pkts >= RULE_PKTS_ALERT or pkts >= 100:
        if numPsh > 5.0: return 1  # mqtt_bruteforce
        return 2  # scan_A
    if pkts <= 30 and meanIat >= 0.1: return 0  # normal
    if meanIat > 0 and meanIat <= 0.01 and pkts > MIN_PKTS_FOR_ML: return 2
    return -1

def train_local_model():
    global X_train_buffer, Y_train_buffer

    # Convertir a numpy
    X = np.array(X_train_buffer, dtype=np.float32)
    Y = np.array(Y_train_buffer, dtype=np.int32)
    
    # Vaciar buffers
    X_train_buffer = []
    Y_train_buffer = []
    
    print(f"\n[ENTRENAMIENTO LOCAL] Iniciando fit sobre {len(X)} muestras incrustadas en Dataset Local...")
    
    # Entrenar modelo y capturar historial (para Analytics)
    hist = model.fit(X, Y, epochs=2, batch_size=8, verbose=1)
    final_acc = float(hist.history.get('accuracy', [0.0])[-1])
    final_loss = float(hist.history.get('loss', [0.0])[-1])
    
    # Extraer pesos buscando explícitamente las capas por su nombre interno
    W3, b3 = model.get_layer("dense_3").get_weights()
    W4, b4 = model.get_layer("dense_out").get_weights()
    
    print(f"[ENTRENAMIENTO LOCAL] Finalizado (Acc: {final_acc:.2%}, Loss: {final_loss:.4f}). Enviando al servidor PC...")
    
    # Enviar al servidor central
    payload = {
        "gateway_id": GATEWAY_ID,
        "num_samples": len(X),
        "round": current_round,
        "accuracy": final_acc,
        "loss": final_loss,
        "W3": W3.tolist(), "b3": b3.tolist(),
        "W4": W4.tolist(), "b4": b4.tolist()
    }
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
        data = json.loads(msg.payload.decode('utf-8'))
        client_id = data.get("client_id", "unknown")
        features = data.get("features", [])
        
        if len(features) != FEATURE_COUNT: return
        
        # Etiquetado Heurístico (Dataset Local Automático)
        label = heuristicLabel(features)
        
        if label >= 0 and features[0] >= MIN_PKTS_FOR_ML:
            X_train_buffer.append(features)
            Y_train_buffer.append(label)
            
            print(f"[DATASET] {client_id} -> Label {CLASS_NAMES[label]} | Buffer: {len(X_train_buffer)}/{SAMPLES_PER_UPDATE}")
            
            if len(X_train_buffer) >= SAMPLES_PER_UPDATE:
                train_local_model()
                
    except Exception as e:
        print(f"[ERROR] {e}")


# ====================== MAIN ======================
if __name__ == "__main__":
    print("=" * 60)
    print(f" GATEWAY HFL v7 [{GATEWAY_ID}] - RASPBERRY PI 4")
    print(f" Entrevista modelos con Keras y agrega en PC")
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
