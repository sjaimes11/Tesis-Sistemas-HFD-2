"""
=============================================================================
 server_hfl.py — PC Servidor Federado (Cloud / Global) — HFL v7 (3 Clases)
=============================================================================
 Topología: PC (Servidor) <-> 2 x Raspberry Pi 4 (Edge Gateways)
 Modelo: 13 -> 32 -> 16 -> 8 -> 3 (softmax)
 Capas FL: W3(16,8) + b3(8) + W4(8,3) + b4(3)
 
 Ejecutar: python server_hfl.py
 Dashboard: http://localhost:8001/
=============================================================================
"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import numpy as np
import requests
import json
import logging
from datetime import datetime
from typing import List, Optional

app = FastAPI(title="Servidor HFL v7 - 3 Clases")

# ====================== ARQUITECTURA ======================
FEATURE_COUNT = 13
L1_UNITS = 32
L2_UNITS = 16
L3_UNITS = 8
OUTPUT_UNITS = 3

CLASS_NAMES = ["normal", "mqtt_bruteforce", "scan_A"]
CLASS_COLORS = ["#2ecc71", "#e74c3c", "#e67e22"]

# ====================== MODELO GLOBAL ======================
W3_global = np.zeros((L2_UNITS, L3_UNITS), dtype=np.float32)
b3_global = np.zeros(L3_UNITS, dtype=np.float32)
W4_global = np.zeros((L3_UNITS, OUTPUT_UNITS), dtype=np.float32)
b4_global = np.zeros(OUTPUT_UNITS, dtype=np.float32)

# ====================== FEDAVG ACUMULADORES ======================
# Acumuladores de las diferentes Raspberrys
W3_update_sum = np.zeros((L2_UNITS, L3_UNITS), dtype=np.float32)
b3_update_sum = np.zeros(L3_UNITS, dtype=np.float32)
W4_update_sum = np.zeros((L3_UNITS, OUTPUT_UNITS), dtype=np.float32)
b4_update_sum = np.zeros(OUTPUT_UNITS, dtype=np.float32)
total_samples_this_round = 0

current_round = 0
updates_received = 0
MIN_UPDATES_PER_ROUND = 2  # Esperamos a las 2 Raspberry Pi 4

history = []
round_in_progress = False

# ====================== IPs de GATEWAYS ======================
# Se puede agregar un registro dinámico, pero por ahora las mantenemos estáticas
# Modifícalas con las IPs reales de tus Raspberry Pi 4
GATEWAYS = [
    "http://192.168.40.120:5000",
    "http://192.168.40.124:5000"
]

class AggregatedModelRequest(BaseModel):
    gateway_id: str
    num_samples: int
    W3: List[List[float]]
    b3: List[float]
    W4: List[List[float]]
    b4: List[float]
    round: int


def distribute_global_model():
    global round_in_progress
    payload = {
        "W3": W3_global.tolist(),
        "b3": b3_global.tolist(),
        "W4": W4_global.tolist(),
        "b4": b4_global.tolist(),
        "round": current_round
    }
    print(f"\n[SERVIDOR] Distribuyendo Modelo Global a Gateways (Ronda {current_round})...")
    
    for gw_url in GATEWAYS:
        try:
            resp = requests.post(f"{gw_url}/deploy-model", json=payload, timeout=5)
            print(f"  -> {gw_url} OK")
        except Exception as e:
            print(f"  -> ERROR publicando a {gw_url}: {e}")

    class_mags = [float(np.mean(np.abs(W4_global[:, j]))) for j in range(OUTPUT_UNITS)]
    history.append({
        "round": current_round,
        "time": datetime.now().strftime("%H:%M:%S"),
        "w3_mag": float(np.mean(np.abs(W3_global))),
        "class_mags": class_mags,
        "event": "deployed"
    })
    round_in_progress = True


@app.post("/aggregate-from-gateway")
def receive_gateway_model(data: AggregatedModelRequest):
    global W3_global, b3_global, W4_global, b4_global
    global W3_update_sum, b3_update_sum, W4_update_sum, b4_update_sum
    global total_samples_this_round, updates_received
    global current_round, round_in_progress

    print(f"\n[SERVIDOR] Pesos recibidos del Gateway '{data.gateway_id}' con {data.num_samples} muestras")

    # Acumulamos sumando ponderadamente por numero de muestras
    W3_np = np.array(data.W3, dtype=np.float32)
    b3_np = np.array(data.b3, dtype=np.float32)
    W4_np = np.array(data.W4, dtype=np.float32)
    b4_np = np.array(data.b4, dtype=np.float32)

    W3_update_sum += W3_np * data.num_samples
    b3_update_sum += b3_np * data.num_samples
    W4_update_sum += W4_np * data.num_samples
    b4_update_sum += b4_np * data.num_samples
    
    total_samples_this_round += data.num_samples
    updates_received += 1

    print(f"  Acumulado: {updates_received} / {MIN_UPDATES_PER_ROUND} gateways")

    # Hacer el FedAvg global cuando superamos el umbral
    if updates_received >= MIN_UPDATES_PER_ROUND:
        current_round += 1
        print(f"\n{'='*60}")
        print(f" FEDAVG GLOBAL - Ronda {current_round} completada con {total_samples_this_round} muestras totales")
        print(f"{'='*60}")
        
        W3_global = W3_update_sum / total_samples_this_round
        b3_global = b3_update_sum / total_samples_this_round
        W4_global = W4_update_sum / total_samples_this_round
        b4_global = b4_update_sum / total_samples_this_round

        # Limpiar buffers
        W3_update_sum.fill(0); b3_update_sum.fill(0)
        W4_update_sum.fill(0); b4_update_sum.fill(0)
        updates_received = 0
        total_samples_this_round = 0
        round_in_progress = False

        # Inmediato despliegue hacia las Raspberrys de nuevo
        distribute_global_model()
        
    return {"status": "ok", "ack_gateway": data.gateway_id}


@app.get("/start-round")
def start_round():
    global current_round
    current_round += 1
    distribute_global_model()
    return {"status": "ok", "message": f"Ronda {current_round} iniciada."}


@app.get("/", response_class=HTMLResponse)
def dashboard():
    class_mags = [float(np.mean(np.abs(W4_global[:, j]))) for j in range(OUTPUT_UNITS)]
    w3_mag = float(np.mean(np.abs(W3_global)))
    max_mag = max(max(class_mags), w3_mag)
    if max_mag == 0: max_mag = 1.0

    status_color = "#f39c12" if round_in_progress else "#2ecc71"
    status_text = f"Esperando Gateways ({updates_received}/{MIN_UPDATES_PER_ROUND})" if round_in_progress else "Global Sync OK"

    # HTML es similar, simplificado por espacio
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Servidor Central HFL v7</title>
    <style>
        body {{ font-family:sans-serif; background:#1a1a2e; color:#eee; padding:20px; }}
        h1 {{ color:#e94560; }}
        .card {{ background:#16213e; padding:15px; border-radius:8px; border:1px solid #0f3460; text-align:center; }}
        .stat {{ font-size: 1.5em; font-weight: bold; color: {status_color}; }}
        .btn {{ padding: 10px 20px; background: #2ecc71; color: white; cursor:pointer; text-decoration:none; border-radius:5px; }}
    </style>
    <script>setTimeout(()=>location.reload(), 5000);</script>
</head>
<body>
    <h1>Servidor HFL v7 - Servidor Central (PC)</h1>
    <p>Topología: Servidor PC <-> 2 Edge Gateways (RPi 4) <-> IoT Nodos (ESP32)</p>
    
    <div style="display:flex; gap: 20px; margin: 20px 0;">
        <div class="card">Ronda Global<br><span class="stat" style="color:#e94560">{current_round}</span></div>
        <div class="card">Gateways Listos<br><span class="stat">{updates_received} / {MIN_UPDATES_PER_ROUND}</span></div>
        <div class="card">Estado<br><span class="stat">{status_text}</span></div>
    </div>
    
    <a href="/start-round" class="btn">🚀 Iniciar Distribuci&oacute;n Forzada</a>
    
    <br><br><h2>Pesos actuales (magnitudes)</h2>
    <p>W3: {w3_mag:.4f} | W4 Normal: {class_mags[0]:.4f} | W4 Bruteforce: {class_mags[1]:.4f} | W4 Scan: {class_mags[2]:.4f}</p>
</body>
</html>"""
    return html

if __name__ == "__main__":
    print("=" * 60)
    print(" SERVIDOR CENTRAL HFL v7 - PC")
    print(" (Recibe pesos de RPi 4, hace FedAvg, distribuye el modelo global)")
    print(" Dashboard: http://localhost:8001/")
    print("=" * 60)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="warning")
