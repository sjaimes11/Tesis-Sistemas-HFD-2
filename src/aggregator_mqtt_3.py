"""
=============================================================================
 aggregator_mqtt.py - Raspberry Pi (Fog Aggregator)
 Modelo binario: 13->32->16->8->1 (normal/attack)
 Escucha MQTT de la ESP32 Broker, hace FedAvg, envía al PC Coordinator
=============================================================================
 Requiere: pip install paho-mqtt numpy requests
 El broker Mosquitto debe estar corriendo: sudo systemctl start mosquitto
=============================================================================
"""
import numpy as np
import requests
import json
import paho.mqtt.client as mqtt

# Arquitectura del modelo (debe coincidir con la ESP32)
# MLP: 13 -> 32 -> 16 -> 8 -> 1
FEATURE_COUNT = 13
L1_UNITS = 32
L2_UNITS = 16
L3_UNITS = 8
OUTPUT_UNITS = 1

CLASS_NAMES = ["normal", "attack"]

# Modelo global base (capas que se comparten via federated learning)
# La ESP32 envia deltas de W3 (16,8), b3 (8), W4 (8,1), b4 (1)
W3_global = np.zeros((L2_UNITS, L3_UNITS), dtype=np.float32)
b3_global = np.zeros(L3_UNITS, dtype=np.float32)
W4_global = np.zeros((L3_UNITS, OUTPUT_UNITS), dtype=np.float32)
b4_global = np.zeros(OUTPUT_UNITS, dtype=np.float32)

# Acumuladores FedAvg
W3_update_sum = np.zeros((L2_UNITS, L3_UNITS), dtype=np.float32)
b3_update_sum = np.zeros(L3_UNITS, dtype=np.float32)
W4_update_sum = np.zeros((L3_UNITS, OUTPUT_UNITS), dtype=np.float32)
b4_update_sum = np.zeros(OUTPUT_UNITS, dtype=np.float32)
total_samples_this_round = 0
updates_received = 0
current_round = 0

MIN_UPDATES_PER_ROUND = 3

# PC Coordinador
IP_TU_PC = "192.168.1.22"
PORT_TU_PC = "8001"
url_coordinador = f"http://{IP_TU_PC}:{PORT_TU_PC}/aggregate-from-pi"

# MQTT - Broker Mosquitto local en la Raspberry
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
TOPIC_UPDATES = "fl/updates"


def on_connect(client, userdata, flags, rc):
    print(f"\n{'='*60}")
    print(f" AGREGADOR MQTT CONECTADO (codigo: {rc})")
    print(f" Modelo binario: 13->32->16->8->1")
    print(f" Capas compartidas: W3({L2_UNITS},{L3_UNITS}) W4({L3_UNITS},{OUTPUT_UNITS})")
    print(f" Coordinador PC: {url_coordinador}")
    print(f"{'='*60}")
    client.subscribe(TOPIC_UPDATES)
    print(f"Suscrito a '{TOPIC_UPDATES}' - esperando ESP32 Broker...\n")


def on_message(client, userdata, msg):
    global W3_update_sum, b3_update_sum
    global W4_update_sum, b4_update_sum
    global total_samples_this_round, updates_received

    try:
        data = json.loads(msg.payload.decode('utf-8'))

        client_id = data.get("client_id", "unknown")
        num_samples = data.get("num_samples", 0)
        round_num = data.get("round", 0)

        print(f"[MQTT] Update de '{client_id}' | Ronda: {round_num} | Muestras: {num_samples}")

        weight_delta = data.get("weight_delta", {})

        # W3: (L2_UNITS, L3_UNITS) = (16, 8)
        delta_W3 = np.array(weight_delta.get("W3", []), dtype=np.float32)
        delta_b3 = np.array(weight_delta.get("b3", []), dtype=np.float32)

        # W4: (L3_UNITS, OUTPUT_UNITS) = (8, 1)
        delta_W4 = np.array(weight_delta.get("W4", []), dtype=np.float32)
        delta_b4 = np.array(weight_delta.get("b4", []), dtype=np.float32)

        # Validar formas
        # W4 llega como array plano [8 valores], reshape a (8,1)
        if delta_W4.ndim == 1 and delta_W4.shape[0] == L3_UNITS:
            delta_W4 = delta_W4.reshape(L3_UNITS, OUTPUT_UNITS)

        errors = []
        if delta_W3.shape != (L2_UNITS, L3_UNITS):
            errors.append(f"W3={delta_W3.shape}, esperado ({L2_UNITS},{L3_UNITS})")
        if delta_b3.shape != (L3_UNITS,):
            errors.append(f"b3={delta_b3.shape}, esperado ({L3_UNITS},)")
        if delta_W4.shape != (L3_UNITS, OUTPUT_UNITS):
            errors.append(f"W4={delta_W4.shape}, esperado ({L3_UNITS},{OUTPUT_UNITS})")
        if delta_b4.shape != (OUTPUT_UNITS,):
            errors.append(f"b4={delta_b4.shape}, esperado ({OUTPUT_UNITS},)")

        if errors:
            for e in errors:
                print(f"  ERROR forma: {e}")
            return

        # Acumular ponderado por num_samples
        W3_update_sum += delta_W3 * num_samples
        b3_update_sum += delta_b3 * num_samples
        W4_update_sum += delta_W4 * num_samples
        b4_update_sum += delta_b4 * num_samples
        total_samples_this_round += num_samples
        updates_received += 1

        print(f"  OK: W3({L2_UNITS},{L3_UNITS}) W4({L3_UNITS},{OUTPUT_UNITS})")
        print(f"  Acumulado: {updates_received}/{MIN_UPDATES_PER_ROUND} updates")

        if updates_received >= MIN_UPDATES_PER_ROUND:
            aggregate_and_send()

    except Exception as e:
        print(f"[ERROR] Procesando mensaje MQTT: {e}")
        import traceback
        traceback.print_exc()


def aggregate_and_send():
    global W3_global, b3_global, W4_global, b4_global
    global W3_update_sum, b3_update_sum, W4_update_sum, b4_update_sum
    global total_samples_this_round, updates_received, current_round

    print(f"\n{'='*60}")
    print(f" FEDAVG - Agregando {updates_received} updates ({total_samples_this_round} muestras)")
    print(f"{'='*60}")

    # FedAvg: promedio ponderado
    W3_avg_delta = W3_update_sum / total_samples_this_round
    b3_avg_delta = b3_update_sum / total_samples_this_round
    W4_avg_delta = W4_update_sum / total_samples_this_round
    b4_avg_delta = b4_update_sum / total_samples_this_round

    W3_global += W3_avg_delta
    b3_global += b3_avg_delta
    W4_global += W4_avg_delta
    b4_global += b4_avg_delta

    current_round += 1

    # Reset acumuladores
    W3_update_sum.fill(0)
    b3_update_sum.fill(0)
    W4_update_sum.fill(0)
    b4_update_sum.fill(0)
    total_samples_this_round = 0
    updates_received = 0

    print(f"Modelo global actualizado (Ronda {current_round})")

    # Magnitud de pesos por capa
    print(f"  W3 mag promedio: {np.mean(np.abs(W3_global)):.6f}")
    print(f"  b3 mag promedio: {np.mean(np.abs(b3_global)):.6f}")
    print(f"  W4 mag promedio: {np.mean(np.abs(W4_global)):.6f}")
    print(f"  b4 mag promedio: {np.mean(np.abs(b4_global)):.6f}")

    # Enviar al PC Coordinador via HTTP
    payload = {
        "round": current_round,
        "model": "binary_13_32_16_8_1",
        "W3": W3_global.tolist(),
        "b3": b3_global.tolist(),
        "W4": W4_global.tolist(),
        "b4": b4_global.tolist(),
    }

    try:
        print(f"\nEnviando a Coordinador PC -> POST {url_coordinador}")
        resp = requests.post(url_coordinador, json=payload, timeout=5)
        print(f"Respuesta PC: {resp.json()}\n")
    except Exception as e:
        print(f"ERROR contactando PC: {e}")
        print("(Se reenviara en la proxima ronda)\n")


if __name__ == "__main__":
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    print("Conectando al broker MQTT local...")
    print(f"Modelo: binario (normal/attack)")
    print(f"Arquitectura: 13->32->16->8->1")
    print(f"Capas compartidas: W3({L2_UNITS},{L3_UNITS}) + W4({L3_UNITS},{OUTPUT_UNITS})")
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except ConnectionRefusedError:
        print(f"ERROR: No se pudo conectar a {MQTT_BROKER}:{MQTT_PORT}")
        print("Asegurate de que Mosquitto esta corriendo:")
        print("  sudo systemctl start mosquitto")
        exit(1)

    client.loop_forever()
