from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np
import requests
import json

app = FastAPI(title="Raspberry Pi - FedAvg Aggregator (Transfer Learning L3)")

NUM_CLASSES = 10
L2_UNITS = 32

# Modelo global base en el Raspberry (Capa 3 solamente para este demo FL)
# W3_global es L2_UNITS x NUM_CLASSES -> (32, 10)
W3_global = np.zeros((L2_UNITS, NUM_CLASSES), dtype=np.float32)
b3_global = np.zeros(NUM_CLASSES, dtype=np.float32)

# Cargamos los pesos base iniciales desde el export para evitar que empiecen en 0
try:
    with open("model_weights.h", "r") as f:
        content = f.read()
    print("WARNING: In a real environment, load true base H5 here. Started global with zeros for now, assuming relative delta updates.")
except:
    pass

# Acumuladores FedAvg
W3_update_sum = np.zeros((L2_UNITS, NUM_CLASSES), dtype=np.float32)
b3_update_sum = np.zeros(NUM_CLASSES, dtype=np.float32)
total_samples_this_round = 0
updates_received = 0

MIN_UPDATES_PER_ROUND = 3

class WeightDelta(BaseModel):
    W3: list[list[float]]
    b3: list[float]

class UpdateRequest(BaseModel):
    client_id: str
    round: int
    num_samples: int
    weight_delta: WeightDelta

@app.post("/update")
def receive_update(data: UpdateRequest):
    global W3_update_sum, b3_update_sum, total_samples_this_round, updates_received
    global W3_global, b3_global

    print(f"Update recibido de {data.client_id} con {data.num_samples} muestras.")

    delta_W3 = np.array(data.weight_delta.W3, dtype=np.float32)
    delta_b3 = np.array(data.weight_delta.b3, dtype=np.float32)

    # Acumulacion
    W3_update_sum += delta_W3 * data.num_samples
    b3_update_sum += delta_b3 * data.num_samples
    total_samples_this_round += data.num_samples
    updates_received += 1

    if updates_received >= MIN_UPDATES_PER_ROUND:
        aggregate_and_update()

    return {"status": "ok", "message": "update received"}

def aggregate_and_update():
    global W3_global, b3_global
    global W3_update_sum, b3_update_sum, total_samples_this_round, updates_received

    print("--- Realizando Agregación FedAvg (Solo Capa 3) ---")
    
    W3_avg_delta = W3_update_sum / total_samples_this_round
    b3_avg_delta = b3_update_sum / total_samples_this_round

    W3_global += W3_avg_delta
    b3_global += b3_avg_delta

    W3_update_sum.fill(0)
    b3_update_sum.fill(0)
    total_samples_this_round = 0
    updates_received = 0

    print("Modelo global (Capa 3) actualizado con éxito.")

    # ENVIAR AL COORDINADOR (COMPUTADOR)
    IP_TU_PC = "192.168.40.11" 
    PORT_TU_PC = "8001"
    url_coordinador = f"http://{IP_TU_PC}:{PORT_TU_PC}/aggregate-from-pi"
    
    payload = {
        "W": W3_global.tolist(),
        "b": b3_global.tolist(),
        "round": 1
    }
    
    try:
        print(f"Enviando modelo agregado al Servidor Central en {url_coordinador}...")
        respuesta = requests.post(url_coordinador, json=payload)
        print("Respuesta del PC:", respuesta.json())
    except Exception as e:
        print("Error contactando al servidor externo (PC):", e)

@app.get("/global-model")
def get_global_model():
    return {
        "round": 1, 
        "W3": W3_global.tolist(),
        "b3": b3_global.tolist()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
