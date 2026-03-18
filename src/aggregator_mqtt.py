"""
=============================================================================
 aggregator_mqtt.py - Raspberry Pi (Fog Aggregator)
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

# Modelo de 6 clases, capa 3: (32, 6)
NUM_CLASSES = 6
L2_UNITS = 32

CLASS_NAMES = [
    "benign",
    "ddos_ack_fragmentation",
    "ddos_icmp_flood",
    "ddos_tcp_flood",
    "dos_syn_flood",
    "dos_tcp_flood"
]

# Modelo global base
W3_global = np.zeros((L2_UNITS, NUM_CLASSES), dtype=np.float32)
b3_global = np.zeros(NUM_CLASSES, dtype=np.float32)

# Acumuladores FedAvg
W3_update_sum = np.zeros((L2_UNITS, NUM_CLASSES), dtype=np.float32)
b3_update_sum = np.zeros(NUM_CLASSES, dtype=np.float32)
total_samples_this_round = 0
updates_received = 0
current_round = 0

# Agregar despues de N updates
MIN_UPDATES_PER_ROUND = 3

# PC Coordinador
IP_TU_PC = "192.168.40.44"
PORT_TU_PC = "8001"
url_coordinador = f"http://{IP_TU_PC}:{PORT_TU_PC}/aggregate-from-pi"

# MQTT - Broker Mosquitto local en la Raspberry
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
TOPIC_UPDATES = "fl/updates"

def on_connect(client, userdata, flags, rc):
    print(f"\n{'='*50}")
    print(f" AGREGADOR MQTT CONECTADO (codigo: {rc})")
    print(f" Modelo: {NUM_CLASSES} clases, Capa 3: ({L2_UNITS},{NUM_CLASSES})")
    print(f" Coordinador PC: {url_coordinador}")
    print(f"{'='*50}")
    client.subscribe(TOPIC_UPDATES)
    print(f"Suscrito a '{TOPIC_UPDATES}' - esperando ESP32 Broker...\n")

def on_message(client, userdata, msg):
    global W3_update_sum, b3_update_sum, total_samples_this_round, updates_received

    try:
        data = json.loads(msg.payload.decode('utf-8'))
        
        client_id = data.get("client_id", "unknown")
        num_samples = data.get("num_samples", 0)
        round_num = data.get("round", 0)
        
        print(f"[MQTT] Update de '{client_id}' | Ronda: {round_num} | Muestras: {num_samples}")

        weight_delta = data.get("weight_delta", {})
        delta_W3 = np.array(weight_delta.get("W3", []), dtype=np.float32)
        delta_b3 = np.array(weight_delta.get("b3", []), dtype=np.float32)

        if delta_W3.shape != (L2_UNITS, NUM_CLASSES) or delta_b3.shape != (NUM_CLASSES,):
            print(f"  ERROR: Forma incorrecta W3={delta_W3.shape} b3={delta_b3.shape}")
            print(f"  Se esperaba W3=({L2_UNITS},{NUM_CLASSES}) b3=({NUM_CLASSES},)")
            return

        # Acumular (ponderado por num_samples)
        W3_update_sum += delta_W3 * num_samples
        b3_update_sum += delta_b3 * num_samples
        total_samples_this_round += num_samples
        updates_received += 1

        print(f"  Acumulado: {updates_received}/{MIN_UPDATES_PER_ROUND} updates")

        if updates_received >= MIN_UPDATES_PER_ROUND:
            aggregate_and_send()

    except Exception as e:
        print(f"[ERROR] Procesando mensaje MQTT: {e}")

def aggregate_and_send():
    global W3_global, b3_global
    global W3_update_sum, b3_update_sum, total_samples_this_round, updates_received
    global current_round

    print(f"\n{'='*50}")
    print(f" FEDAVG - Agregando {updates_received} updates ({total_samples_this_round} muestras)")
    print(f"{'='*50}")
    
    # FedAvg: promedio ponderado
    W3_avg_delta = W3_update_sum / total_samples_this_round
    b3_avg_delta = b3_update_sum / total_samples_this_round

    W3_global += W3_avg_delta
    b3_global += b3_avg_delta

    current_round += 1

    # Reset acumuladores
    W3_update_sum.fill(0)
    b3_update_sum.fill(0)
    total_samples_this_round = 0
    updates_received = 0

    print(f"Modelo global actualizado (Ronda {current_round})")
    
    # Mostrar magnitud de pesos por clase
    for i, name in enumerate(CLASS_NAMES):
        mag = np.mean(np.abs(W3_global[:, i]))
        print(f"  {name}: {mag:.6f}")

    # Enviar al PC Coordinador via HTTP
    payload = {
        "W": W3_global.tolist(),
        "b": b3_global.tolist(),
        "round": current_round
    }
    
    try:
        print(f"\nEnviando a Coordinador PC -> POST {url_coordinador}")
        resp = requests.post(url_coordinador, json=payload, timeout=5)
        print(f"Respuesta PC: {resp.json()}\n")
    except Exception as e:
        print(f"ERROR contactando PC: {e}")
        print("(La Raspberry seguira acumulando, se reenviara en la proxima ronda)\n")

if __name__ == "__main__":
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    print("Conectando al broker MQTT local...")
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except ConnectionRefusedError:
        print(f"ERROR: No se pudo conectar a {MQTT_BROKER}:{MQTT_PORT}")
        print("Asegurate de que Mosquitto esta corriendo:")
        print("  sudo systemctl start mosquitto")
        exit(1)

    client.loop_forever()
