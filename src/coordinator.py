"""
=============================================================================
 coordinator.py - PC Coordinador Central (Cloud)
 Recibe modelo agregado de la Raspberry Pi y muestra dashboard
=============================================================================
 Requiere: pip install fastapi uvicorn numpy
 Correr:   python coordinator.py
 Dashboard: http://localhost:8001/
=============================================================================
"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import numpy as np
from datetime import datetime

app = FastAPI(title="Coordinador FL - Binario")

NUM_CLASSES = 2
L2_UNITS = 16
L3_UNITS = 8

CLASS_NAMES = [
    "normal",
    "attack"
]

CLASS_COLORS = [
    "#2ecc71",  # normal - verde
    "#e74c3c",  # attack - rojo
]

W3_global = np.zeros((L2_UNITS, L3_UNITS), dtype=np.float32)
b3_global = np.zeros(L3_UNITS, dtype=np.float32)
W4_global = np.zeros((L3_UNITS, 1), dtype=np.float32)
b4_global = np.zeros(1, dtype=np.float32)

current_round = 0
updates_count = 0
history = []  # Historial de rondas

class GlobalModelRequest(BaseModel):
    W3: list[list[float]]
    b3: list[float]
    W4: list[list[float]]
    b4: list[float]
    round: int
    model: str = "binary_13_32_16_8_1"

@app.post("/aggregate-from-pi")
def receive_aggregated_model(data: GlobalModelRequest):
    global W3_global, b3_global, W4_global, b4_global, current_round, updates_count

    W3_global = np.array(data.W3, dtype=np.float32)
    b3_global = np.array(data.b3, dtype=np.float32)
    W4_global = np.array(data.W4, dtype=np.float32)
    b4_global = np.array(data.b4, dtype=np.float32)
    current_round = data.round
    updates_count += 1

    # Guardar historial (Solo magnitudes de W3 en este caso para mostrar algo)
    magnitudes = [float(np.mean(np.abs(W3_global[:, i]))) for i in range(min(NUM_CLASSES, L3_UNITS))]
    # Hacemos trampa visual para el dashboard
    mags_dashboard = [float(np.mean(np.abs(W3_global))), float(np.mean(np.abs(W4_global)))]
    
    history.append({
        "round": current_round,
        "time": datetime.now().strftime("%H:%M:%S"),
        "magnitudes": mags_dashboard
    })

    print(f"\n[COORDINADOR] Ronda {current_round} recibida de Raspberry Pi")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name} cap_mag: {mags_dashboard[i]:.6f}")

    return {"status": "ok", "round": current_round}

@app.get("/", response_class=HTMLResponse)
def dashboard():
    # Magnitudes por clase (simplificado para el dashboard)
    mags_dashboard = [float(np.mean(np.abs(W3_global))), float(np.mean(np.abs(W4_global)))] if current_round > 0 else [0.0, 0.0]
    max_mag = max(mags_dashboard) if max(mags_dashboard) > 0 else 1.0

    # Historial JSON para el gráfico
    import json
    hist_json = json.dumps(history[-20:])  # Ultimas 20 rondas

    bars_html = ""
    for i in range(NUM_CLASSES):
        pct = max(2, (mags_dashboard[i] / max_mag) * 100) if max_mag > 0 and current_round > 0 else 0
        bars_html += f"""
        <div class="bar-col">
            <div class="bar-val">{mags_dashboard[i]:.4f}</div>
            <div class="bar" style="height:{pct}%;background:{CLASS_COLORS[i]}"></div>
            <div class="bar-name">{CLASS_NAMES[i].replace('_','<br>')}</div>
        </div>"""

    # Tabla de historial
    rows_html = ""
    for h in reversed(history[-10:]):
        cells = "".join(f"<td>{h['magnitudes'][i]:.5f}</td>" for i in range(NUM_CLASSES))
        rows_html += f"<tr><td><b>{h['round']}</b></td><td>{h['time']}</td>{cells}</tr>"

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard FL - Binario</title>
    <style>
        * {{ margin:0; padding:0; box-sizing:border-box; }}
        body {{ font-family:'Segoe UI',sans-serif; background:#1a1a2e; color:#eee; padding:20px; }}
        .container {{ max-width:1000px; margin:0 auto; }}
        h1 {{ color:#e94560; margin-bottom:5px; font-size:1.8em; }}
        .subtitle {{ color:#888; margin-bottom:20px; }}
        .cards {{ display:grid; grid-template-columns:repeat(3,1fr); gap:15px; margin-bottom:25px; }}
        .card {{ background:#16213e; padding:20px; border-radius:10px; text-align:center; border:1px solid #0f3460; }}
        .card-val {{ font-size:2em; font-weight:bold; color:#e94560; margin-top:5px; }}
        .card-val.online {{ color:#2ecc71; }}
        .section {{ background:#16213e; border-radius:10px; padding:20px; margin-bottom:20px; border:1px solid #0f3460; }}
        .section h2 {{ color:#e94560; margin-bottom:15px; font-size:1.2em; }}
        .chart {{ display:flex; align-items:flex-end; height:180px; gap:8px; padding-bottom:5px; }}
        .bar-col {{ flex:1; display:flex; flex-direction:column; align-items:center; justify-content:flex-end; height:100%; }}
        .bar {{ width:100%; border-radius:4px 4px 0 0; min-height:2px; transition:height 0.5s; }}
        .bar-val {{ font-size:0.7em; color:#aaa; margin-bottom:4px; }}
        .bar-name {{ font-size:0.65em; color:#aaa; margin-top:6px; text-align:center; line-height:1.2; }}
        table {{ width:100%; border-collapse:collapse; font-size:0.85em; }}
        th {{ background:#0f3460; padding:8px; text-align:left; }}
        td {{ padding:6px 8px; border-bottom:1px solid #1a1a3e; }}
        tr:hover {{ background:#1a1a3e; }}
        .refresh {{ color:#888; font-size:0.8em; text-align:right; margin-top:10px; }}
    </style>
    <script>setTimeout(()=>location.reload(), 3000);</script>
</head>
<body>
<div class="container">
    <h1>Monitor Federated Learning</h1>
    <p class="subtitle">Binario | MLP 13→32→16→8→1 | ESP32 → Raspberry Pi → PC</p>
    
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
            <div class="card-val online">Online</div>
        </div>
    </div>

    <div class="section">
        <h2>Magnitud Promedio de Pesos Ocultos vs Salida</h2>
        <div class="chart">{bars_html}</div>
    </div>

    <div class="section">
        <h2>Historial de Rondas</h2>
        <table>
            <tr>
                <th>Ronda</th>
                <th>Hora</th>
                {"".join(f'<th style="color:{CLASS_COLORS[i]}">{CLASS_NAMES[i][:12]}</th>' for i in range(NUM_CLASSES))}
            </tr>
            {rows_html}
        </table>
    </div>
    
    <div class="refresh">Auto-refresh cada 3s</div>
</div>
</body>
</html>"""
    return html

import logging

if __name__ == "__main__":
    print("="*50)
    print(" COORDINADOR FL - PC (Binario)")
    print(f" Dashboard: http://localhost:8001/")
    print("="*50)
    
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="warning")