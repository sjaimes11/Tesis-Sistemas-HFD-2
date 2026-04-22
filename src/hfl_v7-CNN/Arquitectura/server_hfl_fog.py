"""
=============================================================================
 server_hfl_fog.py — PC Servidor Federado — HFL v7-CNN + Fog Layer
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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
import requests
import json
import logging
import csv as csv_module
import base64
import time
import os
import atexit
from datetime import datetime
import threading
from pathlib import Path
import ascon_metrics
from ascon128 import encrypt as ascon_encrypt, decrypt as ascon_decrypt, generate_nonce
from ascon_metrics import AsconMetrics

_counter_lock = threading.Lock()

app = FastAPI(title="Servidor HFL v7 + Fog - Analytics Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR = Path(__file__).resolve().parent / "Results_FOG"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ascon_metrics.RESULTS_DIR = RESULTS_DIR
metrics = AsconMetrics("server_fog")

# ====================== ARQUITECTURA CNN-1D ======================
# Dense1: GAP(16) -> 8   <- capa federada
# Dense_out: 8 -> 3      <- capa federada
FEATURE_COUNT = 13
GAP_OUT      = 16
DENSE1_UNITS = 8
OUTPUT_UNITS = 3
CLASS_NAMES = ["normal", "mqtt_bruteforce", "scan_A"]

# ====================== MODELO GLOBAL ======================
Wd1_global = np.zeros((GAP_OUT,      DENSE1_UNITS), dtype=np.float32)
bd1_global = np.zeros(DENSE1_UNITS,               dtype=np.float32)
Wdo_global = np.zeros((DENSE1_UNITS, OUTPUT_UNITS), dtype=np.float32)
bdo_global = np.zeros(OUTPUT_UNITS,               dtype=np.float32)

# ====================== FEDAVG ACUMULADORES ======================
Wd1_update_sum = np.zeros((GAP_OUT,      DENSE1_UNITS), dtype=np.float32)
bd1_update_sum = np.zeros(DENSE1_UNITS,               dtype=np.float32)
Wdo_update_sum = np.zeros((DENSE1_UNITS, OUTPUT_UNITS), dtype=np.float32)
bdo_update_sum = np.zeros(OUTPUT_UNITS,               dtype=np.float32)

accuracy_sum = 0.0
loss_sum = 0.0
total_samples_this_round = 0
current_round = 0
updates_received = 0

# Con arquitectura Fog, el servidor recibe de fog clusters (no gateways individuales).
# Si hay 1 solo cluster Fog (2 RPis pre-agregando), MIN_UPDATES = 1.
# Si hubiera 2 clusters Fog independientes, MIN_UPDATES = 2.
def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_list(name: str, default: list[str]) -> list[str]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


SERVER_PORT = _env_int("SERVER_PORT", 8001)
MIN_UPDATES_PER_ROUND = _env_int("MIN_UPDATES_PER_ROUND", 1)

history = []
round_in_progress = True

GLOBAL_HISTORY_COLUMNS = [
    "round",
    "time",
    "accuracy",
    "loss",
    "w3_mag",
    "w4_normal",
    "w4_brute",
    "w4_scan",
    "fog_samples",
]

# ====================== IPs de FOG LEADERS ======================
FOG_LEADERS = [
    *_env_list("FOG_LEADERS", ["http://192.168.40.120:5000"]),
]


def next_results_csv_path(prefix: str) -> Path:
    indices = []
    for path in RESULTS_DIR.glob(f"{prefix}_*.csv"):
        suffix = path.stem.replace(f"{prefix}_", "")
        if suffix.isdigit():
            indices.append(int(suffix))

    next_index = (max(indices) + 1) if indices else 1
    return RESULTS_DIR / f"{prefix}_{next_index}.csv"


CURRENT_HISTORY_CSV_PATH = next_results_csv_path("global_weights_history")


def history_rows_for_export():
    rows = []
    for row in history:
        rows.append(
            {
                "round": row["round"],
                "time": row["time"],
                "accuracy": round(float(row["accuracy"]), 6),
                "loss": round(float(row["loss"]), 6),
                "w3_mag": round(float(row["w3_mag"]), 6),
                "w4_normal": round(float(row["w4_normal"]), 6),
                "w4_brute": round(float(row["w4_brute"]), 6),
                "w4_scan": round(float(row["w4_scan"]), 6),
                "fog_samples": int(row.get("fog_samples", 0)),
            }
        )
    return rows


def export_global_history_csv() -> Path | None:
    if not history:
        return None

    rows = history_rows_for_export()
    with CURRENT_HISTORY_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv_module.DictWriter(handle, fieldnames=GLOBAL_HISTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return CURRENT_HISTORY_CSV_PATH


atexit.register(export_global_history_csv)


def list_results_csv_files():
    files = [path for path in RESULTS_DIR.glob("*.csv") if path.is_file()]
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return files


def resolve_results_csv(filename: str) -> Path:
    path = (RESULTS_DIR / filename).resolve()
    if path.parent != RESULTS_DIR.resolve() or not path.exists() or not path.is_file():
        raise FileNotFoundError(filename)
    return path

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
        "W_dense1":    Wd1_global.tolist(), "b_dense1":    bd1_global.tolist(),
        "W_dense_out": Wdo_global.tolist(), "b_dense_out": bdo_global.tolist(),
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
    global Wd1_global, bd1_global, Wdo_global, bdo_global
    global Wd1_update_sum, bd1_update_sum, Wdo_update_sum, bdo_update_sum
    global accuracy_sum, loss_sum
    global total_samples_this_round, updates_received
    global current_round, round_in_progress

    data = ascon_decrypt_payload(envelope, "RPi_leader->PC", current_round)
    if data is None:
        print("[ERROR] ASCON: Tag inválido desde Fog Leader. Mensaje rechazado.")
        return JSONResponse(status_code=403, content={"error": "Invalid ASCON tag"})

    fog_id      = data["gateway_id"]
    num_samples = data["num_samples"]
    accuracy    = data.get("accuracy", 0.0)
    loss        = data.get("loss", 0.0)

    print(f"\n[SERVER-CNN] Pesos FOG-AGREGADOS recibidos de '{fog_id}' "
          f"| {num_samples} muestras | Acc: {accuracy:.2%}")

    Wd1_np = np.array(data["W_dense1"],   dtype=np.float32)
    bd1_np = np.array(data["b_dense1"],   dtype=np.float32)
    Wdo_np = np.array(data["W_dense_out"], dtype=np.float32)
    bdo_np = np.array(data["b_dense_out"], dtype=np.float32)

    Wd1_update_sum += Wd1_np * num_samples
    bd1_update_sum += bd1_np * num_samples
    Wdo_update_sum += Wdo_np * num_samples
    bdo_update_sum += bdo_np * num_samples

    accuracy_sum += accuracy * num_samples
    loss_sum     += loss     * num_samples
    total_samples_this_round += num_samples
    updates_received         += 1

    print(f"  Acumulado: {updates_received} / {MIN_UPDATES_PER_ROUND} fog clusters")

    if updates_received >= MIN_UPDATES_PER_ROUND:
        current_round += 1

        Wd1_global = Wd1_update_sum / total_samples_this_round
        bd1_global = bd1_update_sum / total_samples_this_round
        Wdo_global = Wdo_update_sum / total_samples_this_round
        bdo_global = bdo_update_sum / total_samples_this_round

        acc_global  = accuracy_sum / total_samples_this_round
        loss_global = loss_sum     / total_samples_this_round

        print(f"\n{'='*60}")
        print(f" GLOBAL FEDAVG CNN - Ronda {current_round} ({total_samples_this_round} muestras)")
        print(f" Arquitectura: {len(FOG_LEADERS)} fog cluster(s) con pre-agregación")
        print(f" Global Accuracy: {acc_global:.2%} | Global Loss: {loss_global:.4f}")
        print(f"{'='*60}")

        class_mags = [float(np.mean(np.abs(Wdo_global[:, j]))) for j in range(OUTPUT_UNITS)]

        history.append({
            "round":      current_round,
            "time":       datetime.now().strftime("%H:%M:%S"),
            "accuracy":   float(acc_global),
            "loss":       float(loss_global),
            "w3_mag":     float(np.mean(np.abs(Wd1_global))),  # dense_1 (equiv. W3)
            "w4_normal":  class_mags[0],
            "w4_brute":   class_mags[1],
            "w4_scan":    class_mags[2],
            "fog_samples": total_samples_this_round,
        })

        Wd1_update_sum.fill(0); bd1_update_sum.fill(0)
        Wdo_update_sum.fill(0); bdo_update_sum.fill(0)
        accuracy_sum = 0.0; loss_sum = 0.0
        updates_received         = 0
        total_samples_this_round = 0
        round_in_progress        = False

        export_global_history_csv()
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


@app.get("/api/history/export")
def export_history():
    csv_path = export_global_history_csv()
    if csv_path is None:
        return JSONResponse(status_code=404, content={"error": "Todavia no hay rondas para exportar"})

    return FileResponse(
        path=csv_path,
        media_type="text/csv",
        filename=csv_path.name,
    )


@app.get("/api/results-files")
def get_results_files():
    files = list_results_csv_files()
    return {
        "files": [
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            }
            for path in files
        ]
    }


@app.get("/api/results-csv/{filename}")
def get_results_csv(filename: str):
    try:
        csv_path = resolve_results_csv(filename)
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": "CSV no encontrado"})

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv_module.DictReader(handle))

    return {
        "filename": csv_path.name,
        "rows": rows,
    }


# ====================== DASHBOARD ======================
@app.get("/", response_class=HTMLResponse)
def dashboard():
    html = """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HFL Fog Analytics Dashboard</title>
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
        .csv-controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
            margin-top: 8px;
            margin-bottom: 20px;
        }
        .csv-control-group {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .csv-label {
            color: var(--text-muted);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .csv-input {
            background: #0f172a;
            color: var(--text-main);
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 12px;
        }
        .csv-info {
            color: var(--text-muted);
            margin: 6px 0 0 0;
            font-size: 0.9rem;
        }
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
            <h1 class="title">🚀 Federated IDS Analytics - FOG</h1>
            <p style="color: var(--purple); margin: 4px 0 0 0; font-size: 0.85rem;">☁️ Edge → Fog → Cloud | 🔒 ASCON-128 | Resultados en Results_FOG</p>
        </div>
        <button class="button" onclick="startRound()">Forzar Sincronización Global</button>
    </div>

    <div class="grid">
        <div class="card">
            <div class="card-title">Estado de Red Federada</div>
            <div class="card-value" style="font-size: 1.2rem; margin-top:20px; display:flex; align-items:center; justify-content:center;">
                <span id="ui-dot" class="status-dot dot-orange"></span>
                <span id="ui-status">Esperando Fog Clusters...</span>
            </div>
        </div>
        <div class="card">
            <div class="card-title">Ronda Global</div>
            <div class="card-value" id="ui-round" style="color: var(--accent);">0</div>
        </div>
        <div class="card">
            <div class="card-title">Fog Clusters / Aggregation</div>
            <div class="card-value" id="ui-fog">0 / 1</div>
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
        <div style="display:flex; justify-content:space-between; align-items:center; gap:16px; margin-bottom: 15px;">
            <div class="card-title" style="text-align:left; margin-bottom: 0; color:white; font-size: 1rem;">Historial Dinámico de Pesos Globales</div>
            <div style="display:flex; gap:12px; flex-wrap:wrap;">
                <button class="button" onclick="exportHistoryCsv()">Exportar historial actual CSV</button>
                <span id="history-export-info" class="csv-info" style="margin:0;">Autosave en Results_FOG por cada ronda completada.</span>
            </div>
        </div>
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
                    <th>Fog Samples</th>
                </tr>
            </thead>
            <tbody id="table-body"></tbody>
        </table>
    </div>

    <div class="table-container" style="margin-top: 20px;">
        <div class="card-title" style="text-align:left; margin-bottom: 15px; color:white; font-size: 1rem;">Explorador de CSV FOG</div>
        <div class="csv-controls">
            <div class="csv-control-group">
                <label class="csv-label" for="csv-file-input">Cargar CSV local</label>
                <input id="csv-file-input" class="csv-input" type="file" accept=".csv" />
                <button class="button" onclick="loadLocalCsv()">Visualizar CSV local</button>
            </div>
            <div class="csv-control-group">
                <label class="csv-label" for="results-file-select">Abrir desde Results_FOG</label>
                <select id="results-file-select" class="csv-input"></select>
                <button class="button" onclick="loadSelectedResultsFile()">Visualizar CSV guardado</button>
            </div>
        </div>
        <p id="csv-info" class="csv-info">Selecciona un CSV local o uno guardado en la carpeta Results_FOG.</p>
        <div class="chart-box" style="margin-bottom: 20px;">
            <canvas id="csvMetricsChart"></canvas>
        </div>
        <table>
            <thead id="csv-preview-head"></thead>
            <tbody id="csv-preview-body"></tbody>
        </table>
    </div>

    <script>
        Chart.defaults.color = '#94a3b8';
        Chart.defaults.font.family = 'Inter';

        const ctxAcc = document.getElementById('accChart').getContext('2d');
        const accChart = new Chart(ctxAcc, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Global Accuracy', data: [], borderColor: '#10b981', backgroundColor: 'rgba(16, 185, 129, 0.1)', borderWidth: 3, tension: 0.4, fill: true, pointRadius: 4 }] },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { labels: { color: '#f8fafc', font: {size: 14} } } },
                scales: {
                    x: {
                        grid:{color:'#334155'},
                        ticks: { color: '#94a3b8' },
                        title: { display: true, text: 'Ronda federada', color: '#f8fafc' }
                    },
                    y: {
                        grid:{color:'#334155'},
                        min: 0,
                        max: 1,
                        ticks: { color: '#94a3b8', callback: v => (v*100).toFixed(0) + '%' },
                        title: { display: true, text: 'Accuracy global', color: '#f8fafc' }
                    }
                }
            }
        });

        const ctxLoss = document.getElementById('lossChart').getContext('2d');
        const lossChart = new Chart(ctxLoss, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Global Loss', data: [], borderColor: '#f43f5e', backgroundColor: 'rgba(244, 63, 94, 0.1)', borderWidth: 3, tension: 0.4, fill: true, pointRadius: 4 }] },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { labels: { color: '#f8fafc', font: {size: 14} } } },
                scales: {
                    x: {
                        grid:{color:'#334155'},
                        ticks: { color: '#94a3b8' },
                        title: { display: true, text: 'Ronda federada', color: '#f8fafc' }
                    },
                    y: {
                        grid:{color:'#334155'},
                        ticks: { color: '#94a3b8' },
                        title: { display: true, text: 'Loss global', color: '#f8fafc' }
                    }
                }
            }
        });

        const ctxCsv = document.getElementById('csvMetricsChart').getContext('2d');
        const csvMetricsChart = new Chart(ctxCsv, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'elapsed_ms', data: [], borderColor: '#38bdf8', backgroundColor: 'rgba(56, 189, 248, 0.15)', borderWidth: 3, tension: 0.25, fill: false, yAxisID: 'y' },
                    { label: 'overhead_bytes / payload_bytes', data: [], borderColor: '#f59e0b', backgroundColor: 'rgba(245, 158, 11, 0.15)', borderWidth: 3, tension: 0.25, fill: false, yAxisID: 'y1' }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { labels: { color: '#f8fafc', font: { size: 14 } } } },
                scales: {
                    x: {
                        grid: { color: '#334155' },
                        ticks: { color: '#94a3b8' },
                        title: { display: true, text: 'Índice de registro / ronda', color: '#f8fafc' }
                    },
                    y: {
                        type: 'linear',
                        position: 'left',
                        grid: { color: '#334155' },
                        ticks: { color: '#94a3b8' },
                        title: { display: true, text: 'Tiempo (ms)', color: '#f8fafc' }
                    },
                    y1: {
                        type: 'linear',
                        position: 'right',
                        grid: { drawOnChartArea: false },
                        ticks: { color: '#f59e0b' },
                        title: { display: true, text: 'Bytes', color: '#f59e0b' }
                    }
                }
            }
        });

        function populateTable(history) {
            const tb = document.getElementById('table-body');
            tb.innerHTML = '';
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
                    <td>${row.fog_samples ?? '-'}</td>
                `;
                tb.appendChild(tr);
            });
        }

        function parseCsvText(text) {
            const lines = text.trim().split(/\\r?\\n/).filter(Boolean);
            if (!lines.length) return [];

            const headers = lines[0].split(',').map(v => v.trim());
            return lines.slice(1).map(line => {
                const cols = line.split(',');
                const row = {};
                headers.forEach((header, index) => {
                    row[header] = (cols[index] ?? '').trim();
                });
                return row;
            });
        }

        function renderCsvPreview(rows, sourceName) {
            const info = document.getElementById('csv-info');
            const head = document.getElementById('csv-preview-head');
            const body = document.getElementById('csv-preview-body');

            if (!rows.length) {
                info.innerText = `No se encontraron filas en ${sourceName}.`;
                head.innerHTML = '';
                body.innerHTML = '';
                csvMetricsChart.data.labels = [];
                csvMetricsChart.data.datasets[0].data = [];
                csvMetricsChart.data.datasets[1].data = [];
                csvMetricsChart.update();
                return;
            }

            const headers = Object.keys(rows[0]);
            info.innerText = `${sourceName}: ${rows.length} filas cargadas.`;
            head.innerHTML = `<tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr>`;
            body.innerHTML = rows.slice(0, 25).map(row => `
                <tr>${headers.map(h => `<td>${row[h] ?? ''}</td>`).join('')}</tr>
            `).join('');

            const labels = rows.map((row, index) => row.fl_round || row.round || `${index + 1}`);
            const elapsed = rows.map(row => Number(row.elapsed_ms || 0));
            const bytes = rows.map(row => Number(row.overhead_bytes || row.payload_bytes || 0));

            csvMetricsChart.data.labels = labels;
            csvMetricsChart.data.datasets[0].data = elapsed;
            csvMetricsChart.data.datasets[1].data = bytes;
            csvMetricsChart.update();
        }

        async function loadResultsFiles() {
            try {
                const response = await fetch('/api/results-files');
                const payload = await response.json();
                const select = document.getElementById('results-file-select');
                const files = payload.files || [];

                if (!files.length) {
                    select.innerHTML = '<option value="">No hay CSV guardados</option>';
                    return;
                }

                select.innerHTML = files.map(file => `
                    <option value="${file.name}">${file.name} | ${file.modified}</option>
                `).join('');
            } catch (err) {
                console.error('No se pudo cargar el listado de Results_FOG:', err);
            }
        }

        async function loadSelectedResultsFile() {
            const select = document.getElementById('results-file-select');
            const filename = select.value;
            if (!filename) return;

            try {
                const response = await fetch(`/api/results-csv/${encodeURIComponent(filename)}`);
                const payload = await response.json();
                renderCsvPreview(payload.rows || [], filename);
            } catch (err) {
                console.error('No se pudo cargar el CSV guardado:', err);
            }
        }

        function loadLocalCsv() {
            const input = document.getElementById('csv-file-input');
            const file = input.files && input.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = event => {
                const text = event.target.result || '';
                const rows = parseCsvText(text);
                renderCsvPreview(rows, file.name);
            };
            reader.readAsText(file);
        }

        async function fetchDashboard() {
            try {
                const resStatus = await fetch('/api/status');
                const stat = await resStatus.json();
                
                document.getElementById('ui-round').innerText = stat.current_round;
                document.getElementById('ui-fog').innerText = `${stat.updates_received} / ${stat.min_updates}`;
                
                const dot = document.getElementById('ui-dot');
                const txt = document.getElementById('ui-status');
                
                if (stat.round_in_progress) {
                    dot.className = 'status-dot dot-orange';
                    txt.innerText = 'Entrenamiento Activo...';
                } else {
                    dot.className = 'status-dot dot-green';
                    txt.innerText = 'Actualización Global Distribuida';
                }

                const resHist = await fetch('/api/history');
                const dataHist = await resHist.json();
                const hist = dataHist.history;

                if (hist.length > 0) {
                    const labels = hist.map(h => 'R ' + h.round);
                    const accData = hist.map(h => h.accuracy);
                    const lossData = hist.map(h => h.loss);

                    if(accChart.data.labels.length !== labels.length) {
                        accChart.data.labels = labels;
                        accChart.data.datasets[0].data = accData;
                        accChart.update();

                        lossChart.data.labels = labels;
                        lossChart.data.datasets[0].data = lossData;
                        lossChart.update();

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

        async function exportHistoryCsv() {
            const info = document.getElementById('history-export-info');
            try {
                const response = await fetch('/api/history/export');
                if (!response.ok) {
                    const payload = await response.json();
                    info.innerText = payload.error || 'No se pudo exportar el historial.';
                    return;
                }

                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const disposition = response.headers.get('content-disposition') || '';
                const match = disposition.match(/filename="?([^"]+)"?/i);
                const filename = match ? match[1] : 'global_weights_history.csv';
                const anchor = document.createElement('a');
                anchor.href = url;
                anchor.download = filename;
                document.body.appendChild(anchor);
                anchor.click();
                anchor.remove();
                window.URL.revokeObjectURL(url);
                info.innerText = `Historial exportado: ${filename}`;
                loadResultsFiles();
            } catch (err) {
                console.error('No se pudo exportar el historial:', err);
                info.innerText = 'No se pudo exportar el historial.';
            }
        }

        setInterval(fetchDashboard, 2500);
        fetchDashboard();
        loadResultsFiles();
    </script>
</body>
</html>"""
    return html


# ====================== MAIN ======================
if __name__ == "__main__":
    print("=" * 60)
    print(" SERVIDOR CENTRAL FEDERADO HFL v7-CNN + FOG LAYER")
    print(f" Fog Clusters esperados: {MIN_UPDATES_PER_ROUND}")
    print(f" Fog Leaders: {FOG_LEADERS}")
    print(" Seguridad: ASCON-128 Authenticated Encryption")
    print(" Arquitectura: ESP32 → RPi ↔ RPi → PC (3-tier)")
    print(f" Dashboard: http://localhost:{SERVER_PORT}/")
    print("=" * 60)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT, log_level="warning")
