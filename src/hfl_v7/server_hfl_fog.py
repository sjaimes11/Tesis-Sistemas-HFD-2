"""
=============================================================================
 server_hfl_fog.py — PC Servidor Federado — HFL v7 + Fog Layer
=============================================================================
 Servidor central adaptado para la arquitectura con capa Fog.
 Recibe pesos PRE-AGREGADOS de fog clusters (no de gateways individuales).

 Diferencias con server_hfl.py:
   - Endpoint: /aggregate-from-fog (en lugar de /aggregate-from-gateway)
   - MIN_UPDATES_PER_ROUND = número de fog clusters (default 1)
   - Solo distribuye al líder de cada fog cluster
   - Dashboard muestra info de fog clusters

 Flujo:
   RPi_leader ─HTTP(ASCON)→ PC (FedAvg entre clusters) ─HTTP(ASCON)→ RPi_leader
 
 Ejecutar: python server_hfl_fog.py
 Dashboard: http://localhost:8001/
=============================================================================
"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
import requests
import json
import logging
import base64
import time
from datetime import datetime
import threading
from ascon128 import encrypt as ascon_encrypt, decrypt as ascon_decrypt, generate_nonce
from ascon_metrics import AsconMetrics

_counter_lock = threading.Lock()

metrics = AsconMetrics("server_fog")

app = FastAPI(title="Servidor HFL v7 + Fog - Analytics Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================== ARQUITECTURA ======================
FEATURE_COUNT = 13
L2_UNITS = 16
L3_UNITS = 8
OUTPUT_UNITS = 3
CLASS_NAMES = ["normal", "mqtt_bruteforce", "scan_A"]

# ====================== MODELO GLOBAL ======================
W3_global = np.zeros((L2_UNITS, L3_UNITS), dtype=np.float32)
b3_global = np.zeros(L3_UNITS, dtype=np.float32)
W4_global = np.zeros((L3_UNITS, OUTPUT_UNITS), dtype=np.float32)
b4_global = np.zeros(OUTPUT_UNITS, dtype=np.float32)

# ====================== FEDAVG ACUMULADORES ======================
W3_update_sum = np.zeros((L2_UNITS, L3_UNITS), dtype=np.float32)
b3_update_sum = np.zeros(L3_UNITS, dtype=np.float32)
W4_update_sum = np.zeros((L3_UNITS, OUTPUT_UNITS), dtype=np.float32)
b4_update_sum = np.zeros(OUTPUT_UNITS, dtype=np.float32)

accuracy_sum = 0.0
loss_sum = 0.0
total_samples_this_round = 0
current_round = 0
updates_received = 0

# Con arquitectura Fog, el servidor recibe de fog clusters (no gateways individuales).
# Si hay 1 solo cluster Fog (2 RPis pre-agregando), MIN_UPDATES = 1.
# Si hubiera 2 clusters Fog independientes, MIN_UPDATES = 2.
MIN_UPDATES_PER_ROUND = 1

history = []
round_in_progress = True

# ====================== IPs de FOG LEADERS ======================
FOG_LEADERS = [
    "http://192.168.40.120:5000",
]

ASCON_KEY = bytes([0xA1, 0xB2, 0xC3, 0xD4, 0xE5, 0xF6, 0x07, 0x18,
                   0x29, 0x3A, 0x4B, 0x5C, 0x6D, 0x7E, 0x8F, 0x90])
msg_counter = 0

class EncryptedPayload(BaseModel):
    ct: str
    tag: str
    nonce: str


# ====================== FUNCIONES ASCON ======================
def ascon_encrypt_payload(payload_dict, direction, round_num):
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
    envelope_json = json.dumps(envelope).encode('utf-8')
    metrics.record(direction, "encrypt", len(payload_bytes), len(envelope_json), enc_ms, round_num)
    return envelope


def ascon_decrypt_payload(envelope, direction, round_num):
    ct = base64.b64decode(envelope.ct)
    tag = base64.b64decode(envelope.tag)
    nonce = base64.b64decode(envelope.nonce)

    enc_size = len(envelope.ct) + len(envelope.tag) + len(envelope.nonce) + 50
    t0 = time.perf_counter()
    plaintext = ascon_decrypt(ct, ASCON_KEY, nonce, tag)
    dec_ms = (time.perf_counter() - t0) * 1000

    if plaintext is None:
        return None

    metrics.record(direction, "decrypt", len(plaintext), enc_size, dec_ms, round_num)
    return json.loads(plaintext.decode('utf-8'))


# ====================== DISTRIBUCIÓN GLOBAL ======================
def distribute_global_model():
    global round_in_progress

    payload = {
        "W3": W3_global.tolist(), "b3": b3_global.tolist(),
        "W4": W4_global.tolist(), "b4": b4_global.tolist(),
        "round": current_round
    }
    envelope = ascon_encrypt_payload(payload, "PC->RPi_leader", current_round)

    print(f"\n[SERVER] Distribuyendo Modelo Global (ASCON) a Fog Leaders (Ronda {current_round})...")
    for gw_url in FOG_LEADERS:
        try:
            resp = requests.post(f"{gw_url}/deploy-model", json=envelope, timeout=10)
            print(f"  -> {gw_url} OK")
        except Exception as e:
            print(f"  -> ERROR publicando a {gw_url}: {e}")

    round_in_progress = True


# ====================== ENDPOINT: recibir de Fog cluster ======================
@app.post("/aggregate-from-fog")
async def receive_fog_model(envelope: EncryptedPayload):
    global W3_global, b3_global, W4_global, b4_global
    global W3_update_sum, b3_update_sum, W4_update_sum, b4_update_sum
    global accuracy_sum, loss_sum
    global total_samples_this_round, updates_received
    global current_round, round_in_progress

    data = ascon_decrypt_payload(envelope, "RPi_leader->PC", current_round)
    if data is None:
        print("[ERROR] ASCON: Tag inválido desde Fog Leader. Mensaje rechazado.")
        return JSONResponse(status_code=403, content={"error": "Invalid ASCON tag"})

    fog_id = data["gateway_id"]
    num_samples = data["num_samples"]
    accuracy = data.get("accuracy", 0.0)
    loss = data.get("loss", 0.0)

    print(f"\n[SERVER] Pesos FOG-AGREGADOS recibidos de '{fog_id}' "
          f"| {num_samples} muestras | Acc: {accuracy:.2%}")

    W3_np = np.array(data["W3"], dtype=np.float32)
    b3_np = np.array(data["b3"], dtype=np.float32)
    W4_np = np.array(data["W4"], dtype=np.float32)
    b4_np = np.array(data["b4"], dtype=np.float32)

    W3_update_sum += W3_np * num_samples
    b3_update_sum += b3_np * num_samples
    W4_update_sum += W4_np * num_samples
    b4_update_sum += b4_np * num_samples

    accuracy_sum += accuracy * num_samples
    loss_sum += loss * num_samples
    total_samples_this_round += num_samples
    updates_received += 1

    print(f"  Acumulado: {updates_received} / {MIN_UPDATES_PER_ROUND} fog clusters")

    if updates_received >= MIN_UPDATES_PER_ROUND:
        current_round += 1

        W3_global = W3_update_sum / total_samples_this_round
        b3_global = b3_update_sum / total_samples_this_round
        W4_global = W4_update_sum / total_samples_this_round
        b4_global = b4_update_sum / total_samples_this_round

        acc_global = accuracy_sum / total_samples_this_round
        loss_global = loss_sum / total_samples_this_round

        print(f"\n{'='*60}")
        print(f" GLOBAL FEDAVG - Ronda {current_round} ({total_samples_this_round} muestras)")
        print(f" Arquitectura: {len(FOG_LEADERS)} fog cluster(s) con pre-agregación")
        print(f" Global Accuracy: {acc_global:.2%} | Global Loss: {loss_global:.4f}")
        print(f"{'='*60}")

        class_mags = [float(np.mean(np.abs(W4_global[:, j]))) for j in range(OUTPUT_UNITS)]

        history.append({
            "round": current_round,
            "time": datetime.now().strftime("%H:%M:%S"),
            "accuracy": float(acc_global),
            "loss": float(loss_global),
            "w3_mag": float(np.mean(np.abs(W3_global))),
            "w4_normal": class_mags[0],
            "w4_brute": class_mags[1],
            "w4_scan": class_mags[2],
            "fog_samples": total_samples_this_round
        })

        W3_update_sum.fill(0); b3_update_sum.fill(0)
        W4_update_sum.fill(0); b4_update_sum.fill(0)
        accuracy_sum = 0.0; loss_sum = 0.0
        updates_received = 0
        total_samples_this_round = 0
        round_in_progress = False

        distribute_global_model()
        metrics.print_live_summary()

    return {"status": "ok", "ack_fog": fog_id}


# ====================== API ENDPOINTS ======================
@app.get("/start-round")
def start_round():
    distribute_global_model()
    return {"status": "ok"}


@app.get("/api/status")
def get_status():
    return {
        "round_in_progress": round_in_progress,
        "current_round": current_round,
        "updates_received": updates_received,
        "min_updates": MIN_UPDATES_PER_ROUND,
        "class_names": CLASS_NAMES,
        "fog_leaders": len(FOG_LEADERS)
    }


@app.get("/api/history")
def get_history():
    return {"history": history}


# ====================== DASHBOARD ======================
@app.get("/", response_class=HTMLResponse)
def dashboard():
    html = """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HFL + Fog Analytics Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f172a;
            --panel-bg: #1e293b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #38bdf8;
            --success: #10b981;
            --danger: #f43f5e;
            --warning: #f59e0b;
            --purple: #a78bfa;
        }
        body { font-family: 'Inter', sans-serif; background: var(--bg-color); color: var(--text-main); margin: 0; padding: 30px; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        .title { margin: 0; font-size: 2rem; font-weight: 800; color: var(--accent); }
        .subtitle { color: var(--purple); margin: 4px 0 0 0; font-size: 0.85rem; font-weight: 600; }
        .button {
            background: var(--accent); color: #fff; border: none; padding: 12px 24px;
            border-radius: 8px; cursor: pointer; font-weight: 600; transition: all 0.2s;
            box-shadow: 0 4px 10px rgba(56, 189, 248, 0.4);
        }
        .button:hover { filter: brightness(1.1); transform: translateY(-2px); }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .card { background: var(--panel-bg); padding: 20px; border-radius: 12px; text-align: center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2); }
        .card-title { color: var(--text-muted); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }
        .card-value { font-size: 2rem; font-weight: 800; }
        .charts-container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
        .chart-box { background: var(--panel-bg); padding: 20px; border-radius: 12px; height: 350px; }
        .table-container { background: var(--panel-bg); padding: 20px; border-radius: 12px; overflow-x: auto; max-height: 400px; overflow-y: auto; }
        table { width: 100%; border-collapse: collapse; text-align: left; }
        th, td { padding: 14px; border-bottom: 1px solid #334155; }
        th { color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 0.85rem; position: sticky; top: 0; background: var(--panel-bg); z-index: 10; }
        tbody tr:hover { background-color: #334155; }
        .status-dot { height: 14px; width: 14px; border-radius: 50%; display: inline-block; margin-right: 10px; }
        .dot-green { background-color: var(--success); box-shadow: 0 0 12px var(--success); }
        .dot-orange { background-color: var(--warning); box-shadow: 0 0 12px var(--warning); }
        .badge { display: inline-block; padding: 3px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; }
        .badge-fog { background: rgba(167,139,250,0.2); color: var(--purple); }
        .badge-ascon { background: rgba(245,158,11,0.2); color: var(--warning); }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1 class="title">Federated IDS Analytics</h1>
            <p class="subtitle">
                <span class="badge badge-fog">FOG PRE-AGGREGATION</span>
                <span class="badge badge-ascon">ASCON-128</span>
                &nbsp; Edge → Fog → Cloud (3-tier HFL)
            </p>
        </div>
        <button class="button" onclick="startRound()">Forzar Sincronización</button>
    </div>

    <div class="grid">
        <div class="card">
            <div class="card-title">Estado</div>
            <div class="card-value" style="font-size:1.1rem; margin-top:15px; display:flex; align-items:center; justify-content:center;">
                <span id="ui-dot" class="status-dot dot-orange"></span>
                <span id="ui-status">Esperando Fog Clusters...</span>
            </div>
        </div>
        <div class="card">
            <div class="card-title">Ronda Global</div>
            <div class="card-value" id="ui-round" style="color: var(--accent);">0</div>
        </div>
        <div class="card">
            <div class="card-title">Fog Clusters</div>
            <div class="card-value" id="ui-fog">0 / 1</div>
        </div>
        <div class="card">
            <div class="card-title">Arquitectura</div>
            <div class="card-value" style="font-size:0.9rem; color: var(--purple); margin-top:15px;">
                ESP32 → RPi ↔ RPi → PC
            </div>
        </div>
    </div>

    <div class="charts-container">
        <div class="chart-box"><canvas id="accChart"></canvas></div>
        <div class="chart-box"><canvas id="lossChart"></canvas></div>
    </div>

    <div class="table-container">
        <div class="card-title" style="text-align:left; margin-bottom:15px; color:white; font-size:1rem;">
            Historial de Rondas (Fog-Aggregated)
        </div>
        <table>
            <thead>
                <tr>
                    <th>Ronda</th><th>Hora</th><th>Accuracy</th><th>Loss</th>
                    <th>W3 Mag</th><th>W4 Normal</th><th>W4 Brute</th><th>W4 Scan</th><th>Samples</th>
                </tr>
            </thead>
            <tbody id="table-body"></tbody>
        </table>
    </div>

    <script>
        Chart.defaults.color = '#94a3b8';
        Chart.defaults.font.family = 'Inter';

        const ctxAcc = document.getElementById('accChart').getContext('2d');
        const accChart = new Chart(ctxAcc, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Global Accuracy (Fog)', data: [], borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.1)', borderWidth: 3, tension: 0.4, fill: true, pointRadius: 4 }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#f8fafc', font: {size:14} } } }, scales: { x: { grid:{color:'#334155'} }, y: { grid:{color:'#334155'}, min:0, max:1, ticks: { callback: v => (v*100).toFixed(0)+'%' } } } }
        });

        const ctxLoss = document.getElementById('lossChart').getContext('2d');
        const lossChart = new Chart(ctxLoss, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Global Loss (Fog)', data: [], borderColor: '#f43f5e', backgroundColor: 'rgba(244,63,94,0.1)', borderWidth: 3, tension: 0.4, fill: true, pointRadius: 4 }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#f8fafc', font: {size:14} } } }, scales: { x: { grid:{color:'#334155'} }, y: { grid:{color:'#334155'} } } }
        });

        function populateTable(history) {
            const tb = document.getElementById('table-body');
            tb.innerHTML = '';
            [...history].reverse().forEach((row, i) => {
                const isNew = i === 0 ? 'background-color:rgba(167,139,250,0.1);' : '';
                const tr = document.createElement('tr');
                tr.style = isNew;
                tr.innerHTML = `
                    <td style="color:var(--accent);font-weight:bold;">#${row.round}</td>
                    <td style="color:#94a3b8;">${row.time}</td>
                    <td style="color:var(--success);font-weight:800;">${(row.accuracy*100).toFixed(2)}%</td>
                    <td style="color:var(--danger);font-weight:800;">${row.loss.toFixed(4)}</td>
                    <td>${row.w3_mag.toFixed(5)}</td>
                    <td>${row.w4_normal.toFixed(5)}</td>
                    <td>${row.w4_brute.toFixed(5)}</td>
                    <td>${row.w4_scan.toFixed(5)}</td>
                    <td style="color:var(--purple);">${row.fog_samples || '-'}</td>`;
                tb.appendChild(tr);
            });
        }

        async function fetchDashboard() {
            try {
                const stat = await (await fetch('/api/status')).json();
                document.getElementById('ui-round').innerText = stat.current_round;
                document.getElementById('ui-fog').innerText = `${stat.updates_received} / ${stat.min_updates}`;
                const dot = document.getElementById('ui-dot');
                const txt = document.getElementById('ui-status');
                if (stat.round_in_progress) { dot.className='status-dot dot-orange'; txt.innerText='Entrenamiento Activo...'; }
                else { dot.className='status-dot dot-green'; txt.innerText='Modelo Global Distribuido'; }

                const hist = (await (await fetch('/api/history')).json()).history;
                if (hist.length > 0 && accChart.data.labels.length !== hist.length) {
                    accChart.data.labels = hist.map(h=>'R '+h.round);
                    accChart.data.datasets[0].data = hist.map(h=>h.accuracy);
                    accChart.update();
                    lossChart.data.labels = hist.map(h=>'R '+h.round);
                    lossChart.data.datasets[0].data = hist.map(h=>h.loss);
                    lossChart.update();
                    populateTable(hist);
                }
            } catch(e) { console.error('Dashboard error:', e); }
        }

        function startRound() { fetch('/start-round').then(()=>fetchDashboard()); }
        setInterval(fetchDashboard, 2500);
        fetchDashboard();
    </script>
</body>
</html>"""
    return html


# ====================== MAIN ======================
if __name__ == "__main__":
    print("=" * 60)
    print(" SERVIDOR CENTRAL FEDERADO HFL v7 + FOG LAYER")
    print(f" Fog Clusters esperados: {MIN_UPDATES_PER_ROUND}")
    print(f" Fog Leaders: {FOG_LEADERS}")
    print(" Seguridad: ASCON-128 Authenticated Encryption")
    print(" Arquitectura: ESP32 → RPi ↔ RPi → PC (3-tier)")
    print(f" Dashboard: http://localhost:8001/")
    print("=" * 60)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="warning")
