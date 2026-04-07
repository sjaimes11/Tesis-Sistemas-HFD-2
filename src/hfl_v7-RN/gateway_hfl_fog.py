"""
=============================================================================
 gateway_hfl_fog.py — Raspberry Pi (Fog Gateway) — HFL v7 + Fog Layer
=============================================================================
 Implementa intercomunicación Fog (RPi <-> RPi) vía MQTT con ASCON-128,
 añadiendo una capa de pre-agregación entre gateways antes de enviar al PC.

 Roles (configurar FOG_ROLE):
   "leader" → coordina agregación Fog, comunica con el servidor PC
   "peer"   → envía pesos al líder Fog, recibe modelo global del líder

 Flujo completo (3 niveles de agregación):
   ESP32 ─MQTT→ RPi_peer ─fog MQTT→ RPi_leader ─HTTP→ PC
   ESP32 ←MQTT─ RPi_peer ←fog MQTT─ RPi_leader ←HTTP─ PC

 Despliegue:
   RPi A (líder):  python gateway_hfl_fog.py   (FOG_ROLE="leader")
   RPi B (peer):   python gateway_hfl_fog.py   (FOG_ROLE="peer")

 Los ESP32 NO necesitan cambios — siguen enviando a fl/features en su
 broker local y recibiendo de fl/global_model.
=============================================================================
 Requiere: pip install paho-mqtt numpy requests tensorflow
=============================================================================
"""
import numpy as np
import requests
import json
import threading
import time
import base64
import os

import paho.mqtt.client as mqtt
from http.server import HTTPServer, BaseHTTPRequestHandler
import tensorflow as tf

from ascon128 import encrypt as ascon_encrypt, decrypt as ascon_decrypt, generate_nonce
from ascon_metrics import AsconMetrics

# Lock global para msg_counter (accedido desde hilos MQTT local, MQTT fog e HTTP)
_counter_lock = threading.Lock()

# =============================================================================
#  CONFIGURACIÓN — EDITAR SEGÚN EL ROL DE ESTA RASPBERRY PI
# =============================================================================
FOG_ROLE = os.environ.get("FOG_ROLE", "leader")  # "leader" o "peer"
GATEWAY_ID = os.environ.get("GATEWAY_ID", f"gateway_fog_{FOG_ROLE}")

# IPs — ajustar a tu red
IP_PC = "192.168.40.95"
PORT_PC = "8001"
FOG_LEADER_IP = "192.168.40.120"   # IP de la RPi líder (broker Fog MQTT)
FOG_PEER_IPS = ["192.168.40.124"]  # IPs de las RPi peers (solo el líder las necesita)

url_servidor = f"http://{IP_PC}:{PORT_PC}/aggregate-from-fog"

# MQTT local (para ESP32s de ESTE gateway)
MQTT_LOCAL_BROKER = "localhost"
MQTT_LOCAL_PORT = 1883
TOPIC_FEATURES = "fl/features"
TOPIC_GLOBAL_MODEL = "fl/global_model"

# MQTT Fog (intercomunicación entre gateways)
MQTT_FOG_PORT = 1883
TOPIC_FOG_WEIGHTS = "fog/weights"
TOPIC_FOG_GLOBAL = "fog/global_model"
TOPIC_FOG_READY = "fog/ready"

FOG_EXPECTED_PEERS = 1  # Cuántos peers espera el líder antes de agregar

ASCON_KEY = bytes([0xA1, 0xB2, 0xC3, 0xD4, 0xE5, 0xF6, 0x07, 0x18,
                   0x29, 0x3A, 0x4B, 0x5C, 0x6D, 0x7E, 0x8F, 0x90])
msg_counter = 0

metrics = AsconMetrics(f"fog_{FOG_ROLE}")

# =============================================================================
#  DATASET LOCAL Y MODELO
# =============================================================================
FEATURE_COUNT = 13
CLASS_NAMES = ["normal", "mqtt_bruteforce", "scan_A"]
MIN_PKTS_FOR_ML = 1

X_train_buffer = []
Y_train_buffer = []
SAMPLES_PER_UPDATE = 40

current_round = 0
node_stats = {}

# Buffer de pesos Fog (solo el líder lo usa para agregar)
fog_weights_buffer = {}
fog_weights_lock = threading.Lock()

# =============================================================================
#  MODELO KERAS
# =============================================================================
print(f"[FOG-{FOG_ROLE.upper()}] Cargando modelo base ids_3class.keras...")
try:
    model = tf.keras.models.load_model("ids_3class.keras")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(f"[FOG-{FOG_ROLE.upper()}] Modelo cargado exitosamente.")
except Exception as e:
    print(f"[FOG-{FOG_ROLE.upper()}] Creando modelo desde cero: {e}")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(FEATURE_COUNT,), name='dense_0'),
        tf.keras.layers.Dense(16, activation='relu', name='dense_1'),
        tf.keras.layers.Dense(8, activation='relu', name='dense_3'),
        tf.keras.layers.Dense(3, activation='softmax', name='dense_out')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# =============================================================================
#  UTILIDADES ASCON
# =============================================================================
def ascon_encrypt_json(payload_dict, direction, round_num):
    """Cifra un dict JSON con ASCON-128, retorna envelope dict + métricas."""
    global msg_counter
    payload_bytes = json.dumps(payload_dict).encode('utf-8')
    with _counter_lock:
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
    metrics.record(direction, "encrypt", len(payload_bytes), len(envelope_str), enc_ms, round_num)
    return envelope, envelope_str


def ascon_decrypt_json(envelope_raw, direction, round_num):
    """Descifra un envelope (dict o bytes/str) con ASCON-128, retorna dict."""
    if isinstance(envelope_raw, (bytes, bytearray)):
        envelope = json.loads(envelope_raw.decode('utf-8'))
    elif isinstance(envelope_raw, str):
        envelope = json.loads(envelope_raw)
    else:
        envelope = envelope_raw

    ct = base64.b64decode(envelope["ct"])
    tag = base64.b64decode(envelope["tag"])
    nonce = base64.b64decode(envelope["nonce"])

    raw_size = len(json.dumps(envelope))
    t0 = time.perf_counter()
    plaintext = ascon_decrypt(ct, ASCON_KEY, nonce, tag)
    dec_ms = (time.perf_counter() - t0) * 1000

    if plaintext is None:
        print(f"[ERROR] ASCON: Tag inválido ({direction}). Mensaje rechazado.")
        return None

    metrics.record(direction, "decrypt", len(plaintext), raw_size, dec_ms, round_num)
    return json.loads(plaintext.decode('utf-8'))


# =============================================================================
#  HEURÍSTICA DE ETIQUETADO (misma que gateway_hfl.py)
# =============================================================================
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


def update_node_stats(client_id, label):
    if client_id not in node_stats:
        node_stats[client_id] = {"samples": 0, "last_seen": 0, "labels": {0: 0, 1: 0, 2: 0}}
    node_stats[client_id]["samples"] += 1
    node_stats[client_id]["last_seen"] = time.time()
    if label >= 0:
        node_stats[client_id]["labels"][label] = node_stats[client_id]["labels"].get(label, 0) + 1


def print_node_summary():
    now = time.time()
    print(f"\n{'─'*60}")
    print(f" NODOS CONECTADOS ({len(node_stats)} activos) — {GATEWAY_ID}")
    print(f"{'─'*60}")
    for nid, info in node_stats.items():
        age = now - info["last_seen"]
        status = "●" if age < 15 else "○"
        dist = ", ".join(f"{CLASS_NAMES[k]}:{v}" for k, v in info["labels"].items() if v > 0)
        print(f"  {status} {nid:30s} | total={info['samples']:4d} | {dist}")
    print(f"{'─'*60}")


# =============================================================================
#  MQTT CLIENTS
# =============================================================================
mqtt_local = None   # Broker local (ESP32s)
mqtt_fog = None     # Broker Fog (intercomunicación RPi<->RPi)


# ----- MQTT Local: recibe features de ESP32 -----
def on_local_connect(client, userdata, flags, rc):
    print(f"[MQTT-LOCAL] Conectado a broker local (rc={rc})")
    client.subscribe(TOPIC_FEATURES)
    print(f"[MQTT-LOCAL] Suscrito a '{TOPIC_FEATURES}' para ESP32s")


def on_local_message(client, userdata, msg):
    try:
        data = ascon_decrypt_json(msg.payload, "ESP32->RPi", current_round)
        if data is None:
            return

        client_id = data.get("client_id", "unknown")
        features = data.get("features", [])
        if len(features) != FEATURE_COUNT:
            return

        label = heuristicLabel(features)
        update_node_stats(client_id, label)

        if label >= 0 and features[0] >= MIN_PKTS_FOR_ML:
            X_train_buffer.append(features)
            Y_train_buffer.append(label)
            print(f"[DATASET] {client_id} -> {CLASS_NAMES[label]} | "
                  f"Buffer: {len(X_train_buffer)}/{SAMPLES_PER_UPDATE}")

            if len(X_train_buffer) >= SAMPLES_PER_UPDATE:
                print_node_summary()
                train_local_model()
    except Exception as e:
        print(f"[ERROR-LOCAL] {e}")


# ----- MQTT Fog: intercomunicación entre gateways -----
def on_fog_connect(client, userdata, flags, rc):
    broker_label = "local (líder)" if FOG_ROLE == "leader" else f"remoto ({FOG_LEADER_IP})"
    print(f"[MQTT-FOG] Conectado a broker Fog {broker_label} (rc={rc})")

    if FOG_ROLE == "leader":
        client.subscribe(TOPIC_FOG_WEIGHTS)
        print(f"[MQTT-FOG] Líder suscrito a '{TOPIC_FOG_WEIGHTS}' (esperando pesos de peers)")
    else:
        client.subscribe(TOPIC_FOG_GLOBAL)
        print(f"[MQTT-FOG] Peer suscrito a '{TOPIC_FOG_GLOBAL}' (esperando modelo global del líder)")


def on_fog_message(client, userdata, msg):
    global current_round

    if msg.topic == TOPIC_FOG_WEIGHTS and FOG_ROLE == "leader":
        handle_fog_weights_received(msg.payload)

    elif msg.topic == TOPIC_FOG_GLOBAL and FOG_ROLE == "peer":
        handle_fog_global_received(msg.payload)


# =============================================================================
#  LÍDER: recibe pesos de un peer Fog
# =============================================================================
def handle_fog_weights_received(payload):
    data = ascon_decrypt_json(payload, "RPi_peer->RPi_leader", current_round)
    if data is None:
        return

    peer_id = data["gateway_id"]
    print(f"\n[FOG-LEADER] Pesos recibidos del peer '{peer_id}' "
          f"({data['num_samples']} muestras, Acc: {data['accuracy']:.2%})")

    with fog_weights_lock:
        fog_weights_buffer[peer_id] = {
            "W3": np.array(data["W3"], dtype=np.float32),
            "b3": np.array(data["b3"], dtype=np.float32),
            "W4": np.array(data["W4"], dtype=np.float32),
            "b4": np.array(data["b4"], dtype=np.float32),
            "num_samples": data["num_samples"],
            "accuracy": data["accuracy"],
            "loss": data["loss"]
        }
        check_fog_aggregation_ready()


def check_fog_aggregation_ready():
    """Si el líder tiene sus propios pesos + todos los peers, hace Fog FedAvg."""
    if GATEWAY_ID not in fog_weights_buffer:
        return
    peer_count = sum(1 for k in fog_weights_buffer if k != GATEWAY_ID)
    if peer_count < FOG_EXPECTED_PEERS:
        return

    print(f"\n{'='*60}")
    print(f" FOG AGGREGATION — {len(fog_weights_buffer)} gateways listos")
    print(f"{'='*60}")

    fog_fedavg()


# =============================================================================
#  LÍDER: Fog FedAvg + envío al PC
# =============================================================================
def fog_fedavg():
    global current_round

    total_samples = sum(w["num_samples"] for w in fog_weights_buffer.values())
    if total_samples == 0:
        print("[FOG-LEADER] WARNING: total_samples=0, saltando agregación")
        fog_weights_buffer.clear()
        return

    W3_agg = np.zeros((16, 8), dtype=np.float32)
    b3_agg = np.zeros(8, dtype=np.float32)
    W4_agg = np.zeros((8, 3), dtype=np.float32)
    b4_agg = np.zeros(3, dtype=np.float32)
    acc_agg = 0.0
    loss_agg = 0.0

    for gw_id, w in fog_weights_buffer.items():
        n = w["num_samples"]
        W3_agg += w["W3"] * n
        b3_agg += w["b3"] * n
        W4_agg += w["W4"] * n
        b4_agg += w["b4"] * n
        acc_agg += w["accuracy"] * n
        loss_agg += w["loss"] * n
        print(f"  {gw_id}: {n} muestras, Acc={w['accuracy']:.2%}")

    W3_agg /= total_samples
    b3_agg /= total_samples
    W4_agg /= total_samples
    b4_agg /= total_samples
    acc_agg /= total_samples
    loss_agg /= total_samples

    print(f"  Fog FedAvg: {total_samples} muestras totales, "
          f"Acc promedio={acc_agg:.2%}, Loss={loss_agg:.4f}")

    fog_weights_buffer.clear()

    send_fog_aggregated_to_pc(W3_agg, b3_agg, W4_agg, b4_agg,
                               total_samples, acc_agg, loss_agg)


def send_fog_aggregated_to_pc(W3, b3, W4, b4, num_samples, accuracy, loss):
    payload = {
        "gateway_id": f"fog_cluster_{GATEWAY_ID}",
        "num_samples": num_samples,
        "round": current_round,
        "accuracy": accuracy,
        "loss": loss,
        "W3": W3.tolist(), "b3": b3.tolist(),
        "W4": W4.tolist(), "b4": b4.tolist()
    }
    envelope, _ = ascon_encrypt_json(payload, "RPi_leader->PC", current_round)

    print(f"[FOG-LEADER] Enviando pesos Fog-agregados al servidor PC...")
    try:
        resp = requests.post(url_servidor, json=envelope, timeout=10)
        print(f"  -> Respuesta PC: {resp.json()}")
    except Exception as e:
        print(f"  -> ERROR contactando PC: {e}")


# =============================================================================
#  PEER: recibe modelo global del líder Fog
# =============================================================================
def handle_fog_global_received(payload):
    global current_round

    data = ascon_decrypt_json(payload, "RPi_leader->RPi_peer", current_round)
    if data is None:
        return

    current_round = data.get("round", current_round + 1)
    W3 = np.array(data["W3"], dtype=np.float32)
    b3 = np.array(data["b3"], dtype=np.float32)
    W4 = np.array(data["W4"], dtype=np.float32)
    b4 = np.array(data["b4"], dtype=np.float32)

    print(f"\n{'='*60}")
    print(f" [FOG-PEER] MODELO GLOBAL RECIBIDO DEL LÍDER (Ronda {current_round})")
    print(f" ASCON: Descifrado y autenticado OK")
    print(f"{'='*60}")

    dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
    if len(dense_layers) >= 2:
        dense_layers[-2].set_weights([W3, b3])
        dense_layers[-1].set_weights([W4, b4])

    broadcast_model_to_esp32s(W3, b3, W4, b4)


# =============================================================================
#  HTTP SERVER (solo líder — recibe modelo global del PC)
# =============================================================================
class DeployModelHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        global current_round
        if self.path == "/deploy-model":
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = ascon_decrypt_json(body, "PC->RPi_leader", current_round)

            if data is None:
                self.send_response(403)
                self.end_headers()
                return

            current_round = data.get("round", current_round + 1)
            W3 = np.array(data["W3"], dtype=np.float32)
            b3 = np.array(data["b3"], dtype=np.float32)
            W4 = np.array(data["W4"], dtype=np.float32)
            b4 = np.array(data["b4"], dtype=np.float32)

            print(f"\n{'='*60}")
            print(f" [FOG-LEADER] MODELO GLOBAL RECIBIDO DEL PC (Ronda {current_round})")
            print(f"{'='*60}")

            dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
            if len(dense_layers) >= 2:
                dense_layers[-2].set_weights([W3, b3])
                dense_layers[-1].set_weights([W4, b4])

            broadcast_model_to_esp32s(W3, b3, W4, b4)
            broadcast_model_to_fog_peers(W3, b3, W4, b4)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "round": current_round}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


# =============================================================================
#  BROADCAST: modelo global → ESP32s locales (MQTT + ASCON)
# =============================================================================
def broadcast_model_to_esp32s(W3, b3, W4, b4):
    payload = {
        "round": current_round,
        "W3": W3.tolist(), "b3": b3.tolist(),
        "W4": W4.tolist(), "b4": b4.tolist()
    }
    envelope, envelope_str = ascon_encrypt_json(payload, "RPi->ESP32", current_round)

    if mqtt_local and mqtt_local.is_connected():
        mqtt_local.publish(TOPIC_GLOBAL_MODEL, envelope_str, qos=1)
        print(f"[MQTT-LOCAL] Modelo global cifrado publicado para ESP32s ({len(envelope_str)}B)")


# =============================================================================
#  BROADCAST: modelo global → peers Fog (MQTT + ASCON) — solo líder
# =============================================================================
def broadcast_model_to_fog_peers(W3, b3, W4, b4):
    if FOG_ROLE != "leader":
        return

    payload = {
        "round": current_round,
        "W3": W3.tolist(), "b3": b3.tolist(),
        "W4": W4.tolist(), "b4": b4.tolist()
    }
    envelope, envelope_str = ascon_encrypt_json(payload, "RPi_leader->RPi_peer", current_round)

    if mqtt_fog and mqtt_fog.is_connected():
        mqtt_fog.publish(TOPIC_FOG_GLOBAL, envelope_str, qos=1)
        print(f"[MQTT-FOG] Modelo global cifrado publicado para peers Fog ({len(envelope_str)}B)")


# =============================================================================
#  ENTRENAMIENTO LOCAL
# =============================================================================
def train_local_model():
    global X_train_buffer, Y_train_buffer

    X = np.array(X_train_buffer, dtype=np.float32)
    Y = np.array(Y_train_buffer, dtype=np.int32)
    X_train_buffer = []
    Y_train_buffer = []

    print(f"\n[TRAIN] Entrenando localmente con {len(X)} muestras...")
    hist = model.fit(X, Y, epochs=5, batch_size=8, verbose=1)
    final_acc = float(hist.history.get('accuracy', [0.0])[-1])
    final_loss = float(hist.history.get('loss', [0.0])[-1])

    dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
    W3, b3 = dense_layers[-2].get_weights()
    W4, b4 = dense_layers[-1].get_weights()

    print(f"[TRAIN] Finalizado (Acc: {final_acc:.2%}, Loss: {final_loss:.4f})")

    if FOG_ROLE == "leader":
        with fog_weights_lock:
            fog_weights_buffer[GATEWAY_ID] = {
                "W3": W3.copy(), "b3": b3.copy(),
                "W4": W4.copy(), "b4": b4.copy(),
                "num_samples": len(X),
                "accuracy": final_acc,
                "loss": final_loss
            }
            print(f"[FOG-LEADER] Pesos propios almacenados en buffer Fog")
            check_fog_aggregation_ready()

    elif FOG_ROLE == "peer":
        send_weights_to_fog_leader(W3, b3, W4, b4, len(X), final_acc, final_loss)

    metrics.print_live_summary()


def send_weights_to_fog_leader(W3, b3, W4, b4, num_samples, accuracy, loss):
    """Peer: publica pesos locales al broker Fog del líder via MQTT + ASCON."""
    payload = {
        "gateway_id": GATEWAY_ID,
        "num_samples": num_samples,
        "round": current_round,
        "accuracy": accuracy,
        "loss": loss,
        "W3": W3.tolist(), "b3": b3.tolist(),
        "W4": W4.tolist(), "b4": b4.tolist()
    }
    envelope, envelope_str = ascon_encrypt_json(payload, "RPi_peer->RPi_leader", current_round)

    if mqtt_fog and mqtt_fog.is_connected():
        mqtt_fog.publish(TOPIC_FOG_WEIGHTS, envelope_str, qos=1)
        print(f"[MQTT-FOG] Pesos cifrados enviados al líder Fog ({len(envelope_str)}B)")
    else:
        print(f"[ERROR] No hay conexión al broker Fog del líder")


# =============================================================================
#  MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print(f" GATEWAY HFL v7 + FOG [{GATEWAY_ID}]")
    print(f" Rol Fog: {FOG_ROLE.upper()}")
    if FOG_ROLE == "leader":
        print(f" Peers esperados: {FOG_EXPECTED_PEERS}")
        print(f" Servidor PC: {url_servidor}")
    else:
        print(f" Líder Fog: {FOG_LEADER_IP}:{MQTT_FOG_PORT}")
    print(f" Buffer: {SAMPLES_PER_UPDATE} muestras")
    print(f" Seguridad: ASCON-128 en TODOS los canales")
    print(f"   ESP32 <-> RPi (MQTT local)")
    print(f"   RPi <-> RPi   (MQTT Fog)")
    if FOG_ROLE == "leader":
        print(f"   RPi <-> PC    (HTTP)")
    print("=" * 60)

    # --- MQTT Local (ESP32s) ---
    mqtt_local = mqtt.Client(client_id=f"{GATEWAY_ID}_local")
    mqtt_local.on_connect = on_local_connect
    mqtt_local.on_message = on_local_message
    try:
        mqtt_local.connect(MQTT_LOCAL_BROKER, MQTT_LOCAL_PORT, 60)
        mqtt_local.loop_start()
    except ConnectionRefusedError:
        print(f"[ERROR] Mosquitto local no disponible en {MQTT_LOCAL_BROKER}:{MQTT_LOCAL_PORT}")

    # --- MQTT Fog (intercomunicación RPi <-> RPi) ---
    mqtt_fog = mqtt.Client(client_id=f"{GATEWAY_ID}_fog")
    mqtt_fog.on_connect = on_fog_connect
    mqtt_fog.on_message = on_fog_message

    if FOG_ROLE == "leader":
        fog_broker = MQTT_LOCAL_BROKER
    else:
        fog_broker = FOG_LEADER_IP

    fog_connected = False
    for attempt in range(5):
        try:
            mqtt_fog.connect(fog_broker, MQTT_FOG_PORT, 60)
            mqtt_fog.loop_start()
            print(f"[MQTT-FOG] Conectado a broker Fog en {fog_broker}:{MQTT_FOG_PORT}")
            fog_connected = True
            break
        except (ConnectionRefusedError, OSError) as e:
            wait = 3 * (attempt + 1)
            print(f"[MQTT-FOG] Broker Fog no disponible en {fog_broker}:{MQTT_FOG_PORT} "
                  f"(intento {attempt+1}/5, reintentando en {wait}s): {e}")
            time.sleep(wait)

    if not fog_connected:
        print(f"[ERROR] No se pudo conectar al broker Fog tras 5 intentos. "
              f"Asegúrese de que {'Mosquitto local esté activo' if FOG_ROLE == 'leader' else f'el líder ({FOG_LEADER_IP}) esté ejecutándose'}.")

    # --- HTTP Server (solo líder: recibe modelo global del PC) ---
    if FOG_ROLE == "leader":
        server = HTTPServer(("0.0.0.0", 5000), DeployModelHandler)
        print(f"\n[HTTP] Servidor listo en :5000 para recibir modelo global del PC")
        print(f"[MQTT-LOCAL] Esperando features de ESP32s...")
        print(f"[MQTT-FOG] Esperando pesos de peers Fog...\n")
        server.serve_forever()
    else:
        print(f"\n[MQTT-LOCAL] Esperando features de ESP32s...")
        print(f"[MQTT-FOG] Listo para enviar pesos al líder Fog...\n")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[FOG-PEER] Detenido.")
            mqtt_local.loop_stop()
            mqtt_fog.loop_stop()
