"""
=============================================================================
 server_hfl.py — PC Servidor Federado (Cloud / Global) — HFL v7 (3 Clases)
=============================================================================
 Topología: PC (Servidor) <-> 2 x Raspberry Pi 4 (Edge Gateways)
 Modelo: 13 -> 32 -> 16 -> 8 -> 3 (softmax)
 Capas FL: W3(16,8) + b3(8) + W4(8,3) + b4(3)
 SIN ASCON — Rama para medición de tiempos sin cifrado
 
 Ejecutar: python server_hfl.py
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
import time
from datetime import datetime
from typing import List, Optional

app = FastAPI(title="Servidor HFL v7 - Analytics Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================== ARQUITECTURA ======================
FEATURE_COUNT = 13
L1_UNITS = 32
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

# Acumuladores de métricas
accuracy_sum = 0.0
loss_sum = 0.0

total_samples_this_round = 0
current_round = 0
updates_received = 0
MIN_UPDATES_PER_ROUND = 2

history = []
round_in_progress = True

# ====================== IPs de GATEWAYS ======================
GATEWAYS = [
    "http://192.168.40.120:5000",
    "http://192.168.40.124:5000"
]

class GatewayPayload(BaseModel):
    gateway_id: str
    num_samples: int
    round: int = 0
    accuracy: float = 0.0
    loss: float = 0.0
    W3: list
    b3: list
    W4: list
    b4: list


def distribute_global_model():
    global round_in_progress
    payload = {
        "W3": W3_global.tolist(),
        "b3": b3_global.tolist(),
        "W4": W4_global.tolist(),
        "b4": b4_global.tolist(),
        "round": current_round
    }
    payload_json = json.dumps(payload)
    
    print(f"\n[SERVIDOR] Distribuyendo Modelo Global a Gateways (Ronda {current_round}, {len(payload_json)}B, sin cifrado)...")
    
    for gw_url in GATEWAYS:
        try:
            t0 = time.perf_counter()
            resp = requests.post(f"{gw_url}/deploy-model", json=payload, timeout=5)
            send_ms = (time.perf_counter() - t0) * 1000
            print(f"  -> {gw_url} OK ({send_ms:.1f}ms)")
        except Exception as e:
            print(f"  -> ERROR publicando a {gw_url}: {e}")

    round_in_progress = True


@app.post("/aggregate-from-gateway")
async def receive_gateway_model(payload: GatewayPayload):
    global W3_global, b3_global, W4_global, b4_global
    global W3_update_sum, b3_update_sum, W4_update_sum, b4_update_sum
    global accuracy_sum, loss_sum
    global total_samples_this_round, updates_received
    global current_round, round_in_progress

    t0 = time.perf_counter()
    gateway_id = payload.gateway_id
    num_samples = payload.num_samples
    accuracy = payload.accuracy
    loss = payload.loss
    data = payload.dict()
    parse_ms = (time.perf_counter() - t0) * 1000
    
    print(f"\n[SERVIDOR] Pesos recibidos de '{gateway_id}' | {num_samples} muestras | Acc: {accuracy:.2%} | {parse_ms:.3f}ms (sin cifrado)")

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

    print(f"  Acumulado: {updates_received} / {MIN_UPDATES_PER_ROUND} gateways")

    if updates_received >= MIN_UPDATES_PER_ROUND:
        current_round += 1
        
        W3_global = W3_update_sum / total_samples_this_round
        b3_global = b3_update_sum / total_samples_this_round
        W4_global = W4_update_sum / total_samples_this_round
        b4_global = b4_update_sum / total_samples_this_round
        
        acc_global = accuracy_sum / total_samples_this_round
        loss_global = loss_sum / total_samples_this_round

        print(f"\n{'='*60}")
        print(f" FEDAVG GLOBAL - Ronda {current_round} completada ({total_samples_this_round} muestras)")
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
            "w4_scan": class_mags[2]
        })

        W3_update_sum.fill(0); b3_update_sum.fill(0)
        W4_update_sum.fill(0); b4_update_sum.fill(0)
        accuracy_sum = 0.0
        loss_sum = 0.0
        updates_received = 0
        total_samples_this_round = 0
        round_in_progress = False

        distribute_global_model()
        
    return {"status": "ok", "ack_gateway": gateway_id}


@app.get("/start-round")
def start_round():
    global current_round
    # Si presionan forzar, no sumamos ronda pero distribuimos lo que tengamos
    distribute_global_model()
    return {"status": "ok"}


@app.get("/api/status")
def get_status():
    return {
        "round_in_progress": round_in_progress,
        "current_round": current_round,
        "updates_received": updates_received,
        "min_updates": MIN_UPDATES_PER_ROUND,
        "class_names": CLASS_NAMES
    }


@app.get("/api/history")
def get_history():
    return {"history": history}


@app.get("/", response_class=HTMLResponse)
def dashboard():
    html = """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HFL Analytics Dashboard</title>
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
        }
        body { 
            font-family: 'Inter', sans-serif; 
            background: var(--bg-color); 
            color: var(--text-main); 
            margin: 0; 
            padding: 30px; 
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .title { margin: 0; font-size: 2rem; font-weight: 800; color: var(--accent); }
        .button {
            background: var(--accent);
            color: #fff;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.2s;
            box-shadow: 0 4px 10px rgba(56, 189, 248, 0.4);
        }
        .button:hover { filter: brightness(1.1); transform: translateY(-2px); box-shadow: 0 6px 15px rgba(56, 189, 248, 0.6);}
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: var(--panel-bg);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);
        }
        .card-title { color: var(--text-muted); font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;}
        .card-value { font-size: 2.2rem; font-weight: 800; }
        
        .charts-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .chart-box {
            background: var(--panel-bg);
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);
            height: 350px;
        }

        .table-container {
            background: var(--panel-bg);
            padding: 20px;
            border-radius: 12px;
            overflow-x: auto;
            max-height: 400px;
            overflow-y: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }
        th, td { padding: 14px; border-bottom: 1px solid #334155; }
        th { color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 0.85rem; position: sticky; top: 0; background: var(--panel-bg); z-index: 10;}
        tbody tr:hover { background-color: #334155; }
        
        .status-dot {
            height: 14px; width: 14px; border-radius: 50%; display: inline-block; margin-right: 10px;
        }
        .dot-green { background-color: var(--success); box-shadow: 0 0 12px var(--success);}
        .dot-orange { background-color: var(--warning); box-shadow: 0 0 12px var(--warning);}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1 class="title">🚀 Federated IDS Analytics</h1>
            <p style="color: var(--warning); margin: 4px 0 0 0; font-size: 0.85rem;">⚡ SIN CIFRADO — Medición de baseline</p>
        </div>
        <button class="button" onclick="startRound()">Forzar Sincronización Global</button>
    </div>

    <div class="grid">
        <div class="card">
            <div class="card-title">Estado de Red Federada</div>
            <div class="card-value" style="font-size: 1.2rem; margin-top:20px; display:flex; align-items:center; justify-content:center;">
                <span id="ui-dot" class="status-dot dot-orange"></span>
                <span id="ui-status">Esperando Gateways...</span>
            </div>
        </div>
        <div class="card">
            <div class="card-title">Ronda Global</div>
            <div class="card-value" id="ui-round" style="color: var(--accent);">0</div>
        </div>
        <div class="card">
            <div class="card-title">Gateways / Aggregation</div>
            <div class="card-value" id="ui-gateways">0 / 2</div>
        </div>
    </div>

    <div class="charts-container">
        <div class="chart-box">
            <canvas id="accChart"></canvas>
        </div>
        <div class="chart-box">
            <canvas id="lossChart"></canvas>
        </div>
    </div>

    <div class="table-container">
        <div class="card-title" style="text-align:left; margin-bottom: 15px; color:white; font-size: 1rem;">Historial Dinámico de Pesos Globales</div>
        <table>
            <thead>
                <tr>
                    <th>Ronda</th>
                    <th>Hora</th>
                    <th>Global Accuracy</th>
                    <th>Global Loss</th>
                    <th>W3 (General)</th>
                    <th>W4 Normal</th>
                    <th>W4 Bruteforce</th>
                    <th>W4 Scan_A</th>
                </tr>
            </thead>
            <tbody id="table-body">
                <!-- Data is injected here by JS -->
            </tbody>
        </table>
    </div>

    <script>
        Chart.defaults.color = '#94a3b8';
        Chart.defaults.font.family = 'Inter';

        // Inicializar Gráficas
        const ctxAcc = document.getElementById('accChart').getContext('2d');
        const accChart = new Chart(ctxAcc, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Global Accuracy', data: [], borderColor: '#10b981', backgroundColor: 'rgba(16, 185, 129, 0.1)', borderWidth: 3, tension: 0.4, fill: true, pointRadius: 4 }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#f8fafc', font: {size: 14} } } }, scales: { x: { grid:{color:'#334155'}, ticks: { color: '#94a3b8' } }, y: { grid:{color:'#334155'}, min: 0, max: 1, ticks: { color: '#94a3b8', callback: v => (v*100).toFixed(0) + '%' } } } }
        });

        const ctxLoss = document.getElementById('lossChart').getContext('2d');
        const lossChart = new Chart(ctxLoss, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Global Loss', data: [], borderColor: '#f43f5e', backgroundColor: 'rgba(244, 63, 94, 0.1)', borderWidth: 3, tension: 0.4, fill: true, pointRadius: 4 }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#f8fafc', font: {size: 14} } } }, scales: { x: { grid:{color:'#334155'}, ticks: { color: '#94a3b8' } }, y: { grid:{color:'#334155'}, ticks: { color: '#94a3b8' } } } }
        });

        function populateTable(history) {
            const tb = document.getElementById('table-body');
            tb.innerHTML = '';
            // Mostrar inverso para que las más nuevas estén arriba
            const rev = [...history].reverse();
            rev.forEach((row, index) => {
                const isNew = index === 0 ? 'background-color: rgba(56, 189, 248, 0.1);' : '';
                const tr = document.createElement('tr');
                tr.style = isNew;
                tr.innerHTML = `
                    <td style="color:var(--accent); font-weight:bold;">#${row.round}</td>
                    <td style="color:#94a3b8;">${row.time}</td>
                    <td style="color:var(--success); font-weight:800;">${(row.accuracy * 100).toFixed(2)}%</td>
                    <td style="color:var(--danger); font-weight:800;">${row.loss.toFixed(4)}</td>
                    <td>${row.w3_mag.toFixed(5)}</td>
                    <td>${row.w4_normal.toFixed(5)}</td>
                    <td>${row.w4_brute.toFixed(5)}</td>
                    <td>${row.w4_scan.toFixed(5)}</td>
                `;
                tb.appendChild(tr);
            });
        }

        async function fetchDashboard() {
            try {
                const resStatus = await fetch('/api/status');
                const stat = await resStatus.json();
                
                document.getElementById('ui-round').innerText = stat.current_round;
                document.getElementById('ui-gateways').innerText = `${stat.updates_received} / ${stat.min_updates}`;
                
                const dot = document.getElementById('ui-dot');
                const txt = document.getElementById('ui-status');
                
                if (stat.round_in_progress) {
                    dot.className = 'status-dot dot-orange';
                    txt.innerText = 'Entrenamiento Activo...';
                } else {
                    dot.className = 'status-dot dot-green';
                    txt.innerText = 'Actualización Global Distribuida';
                }

                // Fetch histórico para gráficas
                const resHist = await fetch('/api/history');
                const dataHist = await resHist.json();
                const hist = dataHist.history;

                if (hist.length > 0) {
                    const labels = hist.map(h => 'R ' + h.round);
                    const accData = hist.map(h => h.accuracy);
                    const lossData = hist.map(h => h.loss);

                    // Actualizar Chart solo si hay datos nuevos
                    if(accChart.data.labels.length !== labels.length) {
                        accChart.data.labels = labels;
                        accChart.data.datasets[0].data = accData;
                        accChart.update();

                        lossChart.data.labels = labels;
                        lossChart.data.datasets[0].data = lossData;
                        lossChart.update();

                        // Llenar tabla
                        populateTable(hist);
                    }
                }

            } catch (err) {
                console.error("Dashboard fetch error (Server may be down):", err);
            }
        }

        function startRound() {
            fetch('/start-round').then(() => fetchDashboard());
        }

        // Auto-refresh via AJAX sin parpadeos cada 2.5 segundos
        setInterval(fetchDashboard, 2500);
        fetchDashboard();
    </script>
</body>
</html>"""
    return html

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print(" SERVIDOR CENTRAL FEDERADO HFL v7 - ANALYTICS DASHBOARD")
    print(" *** SIN CIFRADO ASCON — Medición de baseline ***")
    print(" -> Dashboard Gráfico: http://localhost:8001/")
    print("=" * 60)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="warning")
