"""
=============================================================================
 coordinator_hfl.py — PC Coordinador Central (Cloud) — HFL Bidireccional
=============================================================================
 Rol: Mantiene el modelo global. Inicia rondas FL enviando pesos al Pi.
      Recibe modelo agregado (FedAvg) del Pi y actualiza el modelo global.
 
 Ejecutar: python coordinator_hfl.py
 Dashboard: http://localhost:8001/
=============================================================================
 Requiere: pip install fastapi uvicorn numpy requests
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

app = FastAPI(title="Coordinador HFL - Binario")

# ====================== ARQUITECTURA DEL MODELO ======================
# MLP: 13 -> 32 -> 16 -> 8 -> 1 (binario: normal/attack)
# Capas compartidas via FL: W3(16,8) + b3(8) + W4(8,1) + b4(1)
FEATURE_COUNT = 13
L1_UNITS = 32
L2_UNITS = 16
L3_UNITS = 8
OUTPUT_UNITS = 1

CLASS_NAMES = ["normal", "attack"]
CLASS_COLORS = ["#2ecc71", "#e74c3c"]

# ====================== MODELO GLOBAL ======================
W3_global = np.zeros((L2_UNITS, L3_UNITS), dtype=np.float32)
b3_global = np.zeros(L3_UNITS, dtype=np.float32)
W4_global = np.zeros((L3_UNITS, OUTPUT_UNITS), dtype=np.float32)
b4_global = np.zeros(OUTPUT_UNITS, dtype=np.float32)

current_round = 0
updates_count = 0
history = []
round_in_progress = False

# ====================== CONFIGURACION RASPBERRY PI ======================
# Cambiar esta IP a la IP actual de tu Raspberry Pi
RASPBERRY_PI_IP = "192.168.1.21"
RASPBERRY_PI_PORT = 5000
url_raspberry = f"http://{RASPBERRY_PI_IP}:{RASPBERRY_PI_PORT}/deploy-model"


# ====================== ENDPOINTS ======================

class AggregatedModelRequest(BaseModel):
    W3: list[list[float]]
    b3: list[float]
    W4: list[list[float]]
    b4: list[float]
    round: int
    model: str = "binary_13_32_16_8_1"


@app.post("/aggregate-from-pi")
def receive_aggregated_model(data: AggregatedModelRequest):
    """Recibe el modelo agregado (FedAvg) de la Raspberry Pi."""
    global W3_global, b3_global, W4_global, b4_global
    global current_round, updates_count, round_in_progress

    W3_global = np.array(data.W3, dtype=np.float32)
    b3_global = np.array(data.b3, dtype=np.float32)
    W4_global = np.array(data.W4, dtype=np.float32)
    b4_global = np.array(data.b4, dtype=np.float32)
    current_round = data.round
    updates_count += 1
    round_in_progress = False  # Ronda terminada

    mags = [float(np.mean(np.abs(W3_global))), float(np.mean(np.abs(W4_global)))]
    history.append({
        "round": current_round,
        "time": datetime.now().strftime("%H:%M:%S"),
        "magnitudes": mags,
        "event": "aggregated"
    })

    print(f"\n[COORDINADOR] Ronda {current_round} COMPLETADA (FedAvg del Pi)")
    print(f"  W3 mag: {mags[0]:.6f}  W4 mag: {mags[1]:.6f}")

    return {"status": "ok", "round": current_round}


@app.get("/start-round")
def start_round():
    """Inicia una nueva ronda de FL: envía el modelo global al Pi."""
    global round_in_progress

    if round_in_progress:
        return {"status": "busy", "message": "Ya hay una ronda en progreso."}

    payload = {
        "W3": W3_global.tolist(),
        "b3": b3_global.tolist(),
        "W4": W4_global.tolist(),
        "b4": b4_global.tolist(),
        "round": current_round + 1
    }

    try:
        print(f"\n[COORDINADOR] Iniciando Ronda {current_round + 1}...")
        print(f"  Enviando modelo global al Pi -> POST {url_raspberry}")
        resp = requests.post(url_raspberry, json=payload, timeout=10)
        result = resp.json()
        round_in_progress = True

        history.append({
            "round": current_round + 1,
            "time": datetime.now().strftime("%H:%M:%S"),
            "magnitudes": [float(np.mean(np.abs(W3_global))), float(np.mean(np.abs(W4_global)))],
            "event": "deployed"
        })

        print(f"  Respuesta Pi: {result}")
        return {"status": "ok", "message": f"Ronda {current_round + 1} iniciada. Modelo enviado al Pi."}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/auto-round")
def auto_round():
    """Inicia una ronda automáticamente (útil para el dashboard)."""
    return start_round()


@app.get("/", response_class=HTMLResponse)
def dashboard():
    mags = [float(np.mean(np.abs(W3_global))), float(np.mean(np.abs(W4_global)))] if current_round > 0 else [0.0, 0.0]
    max_mag = max(mags) if max(mags) > 0 else 1.0

    status_color = "#f39c12" if round_in_progress else "#2ecc71"
    status_text = "Entrenando..." if round_in_progress else "Listo"

    bars_html = ""
    labels = ["W3 (oculta)", "W4 (salida)"]
    for i in range(2):
        pct = max(2, (mags[i] / max_mag) * 100) if max_mag > 0 and current_round > 0 else 0
        bars_html += f"""
        <div class="bar-col">
            <div class="bar-val">{mags[i]:.4f}</div>
            <div class="bar" style="height:{pct}%;background:{CLASS_COLORS[i]}"></div>
            <div class="bar-name">{labels[i]}</div>
        </div>"""

    rows_html = ""
    for h in reversed(history[-15:]):
        ev = h.get("event", "?")
        ev_icon = "🚀" if ev == "deployed" else "✅"
        rows_html += f"""<tr>
            <td><b>{h['round']}</b></td>
            <td>{h['time']}</td>
            <td>{ev_icon} {ev}</td>
            <td>{h['magnitudes'][0]:.5f}</td>
            <td>{h['magnitudes'][1]:.5f}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HFL Coordinator</title>
    <style>
        * {{ margin:0; padding:0; box-sizing:border-box; }}
        body {{ font-family:'Segoe UI',sans-serif; background:#1a1a2e; color:#eee; padding:20px; }}
        .container {{ max-width:1000px; margin:0 auto; }}
        h1 {{ color:#e94560; margin-bottom:5px; font-size:1.8em; }}
        .subtitle {{ color:#888; margin-bottom:20px; }}
        .cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:15px; margin-bottom:25px; }}
        .card {{ background:#16213e; padding:20px; border-radius:10px; text-align:center; border:1px solid #0f3460; }}
        .card-val {{ font-size:1.8em; font-weight:bold; color:#e94560; margin-top:5px; }}
        .card-val.online {{ color:{status_color}; }}
        .section {{ background:#16213e; border-radius:10px; padding:20px; margin-bottom:20px; border:1px solid #0f3460; }}
        .section h2 {{ color:#e94560; margin-bottom:15px; font-size:1.2em; }}
        .chart {{ display:flex; align-items:flex-end; height:180px; gap:20px; padding-bottom:5px; }}
        .bar-col {{ flex:1; display:flex; flex-direction:column; align-items:center; justify-content:flex-end; height:100%; }}
        .bar {{ width:100%; border-radius:4px 4px 0 0; min-height:2px; transition:height 0.5s; }}
        .bar-val {{ font-size:0.7em; color:#aaa; margin-bottom:4px; }}
        .bar-name {{ font-size:0.75em; color:#aaa; margin-top:6px; text-align:center; }}
        table {{ width:100%; border-collapse:collapse; font-size:0.85em; }}
        th {{ background:#0f3460; padding:8px; text-align:left; }}
        td {{ padding:6px 8px; border-bottom:1px solid #1a1a3e; }}
        tr:hover {{ background:#1a1a3e; }}
        .btn {{ display:inline-block; padding:12px 30px; border-radius:8px; font-size:1.1em;
                font-weight:bold; cursor:pointer; border:none; margin:5px; text-decoration:none; }}
        .btn-start {{ background:#2ecc71; color:#fff; }}
        .btn-start:hover {{ background:#27ae60; }}
        .btn-start.disabled {{ background:#555; cursor:not-allowed; }}
        .actions {{ text-align:center; margin-bottom:20px; }}
        .refresh {{ color:#888; font-size:0.8em; text-align:right; margin-top:10px; }}
        .flow {{ background:#0f3460; padding:15px; border-radius:8px; text-align:center;
                 font-size:0.9em; color:#aaa; margin-bottom:20px; letter-spacing:1px; }}
        .flow span {{ color:#e94560; font-weight:bold; }}
    </style>
    <script>setTimeout(()=>location.reload(), 5000);</script>
</head>
<body>
<div class="container">
    <h1>Coordinador HFL</h1>
    <p class="subtitle">Hierarchical Federated Learning | Binario | MLP 13→32→16→8→1</p>

    <div class="flow">
        <span>PC</span> →HTTP→ <span>Raspberry Pi</span> →MQTT→ <span>ESP32 Broker</span> ←MQTT← <span>ESP32 Sensor</span>
    </div>

    <div class="actions">
        <a class="btn btn-start {'disabled' if round_in_progress else ''}"
           href="/start-round"
           {'onclick="return false;"' if round_in_progress else ''}>
            {'⏳ Ronda en progreso...' if round_in_progress else '🚀 Iniciar Ronda FedAvg'}
        </a>
    </div>

    <div class="cards">
        <div class="card">
            <div>Ronda Global</div>
            <div class="card-val">{current_round}</div>
        </div>
        <div class="card">
            <div>Updates Recibidos</div>
            <div class="card-val">{updates_count}</div>
        </div>
        <div class="card">
            <div>Estado</div>
            <div class="card-val online">{status_text}</div>
        </div>
        <div class="card">
            <div>Pi Target</div>
            <div class="card-val" style="font-size:0.8em;color:#3498db">{RASPBERRY_PI_IP}</div>
        </div>
    </div>

    <div class="section">
        <h2>Magnitud Promedio de Pesos Globales</h2>
        <div class="chart">{bars_html}</div>
    </div>

    <div class="section">
        <h2>Historial de Eventos</h2>
        <table>
            <tr><th>Ronda</th><th>Hora</th><th>Evento</th><th style="color:#2ecc71">W3 mag</th><th style="color:#e74c3c">W4 mag</th></tr>
            {rows_html}
        </table>
    </div>

    <div class="refresh">Auto-refresh cada 5s</div>
</div>
</body>
</html>"""
    return html


if __name__ == "__main__":
    print("=" * 60)
    print(" COORDINADOR HFL - PC (Cloud)")
    print(f" Dashboard: http://localhost:8001/")
    print(f" Raspberry Pi target: {url_raspberry}")
    print("=" * 60)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="warning")
