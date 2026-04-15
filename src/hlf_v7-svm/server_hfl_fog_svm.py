import base64
import json
import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from ascon128 import decrypt as ascon_decrypt
from ascon_metrics import AsconMetrics


BASE_DIR = Path(__file__).resolve().parent
ASCON_KEY = bytes([0xA1, 0xB2, 0xC3, 0xD4, 0xE5, 0xF6, 0x07, 0x18,
                   0x29, 0x3A, 0x4B, 0x5C, 0x6D, 0x7E, 0x8F, 0x90])

metrics = AsconMetrics("server_svm")
app = FastAPI(title="server_hfl_svm")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

history = []


class EncryptedPayload(BaseModel):
    ct: str
    tag: str
    nonce: str


def decrypt_payload(envelope: EncryptedPayload):
    ct = base64.b64decode(envelope.ct)
    tag = base64.b64decode(envelope.tag)
    nonce = base64.b64decode(envelope.nonce)

    t0 = time.perf_counter()
    plaintext = ascon_decrypt(ct, ASCON_KEY, nonce, tag)
    dec_ms = (time.perf_counter() - t0) * 1000
    if plaintext is None:
        return None

    metrics.record("RPi->PC", "decrypt", len(plaintext), len(envelope.ct) + len(envelope.tag) + len(envelope.nonce), dec_ms, 0)
    return json.loads(plaintext.decode("utf-8"))


@app.post("/ingest-prediction")
async def ingest_prediction(envelope: EncryptedPayload):
    payload = decrypt_payload(envelope)
    if payload is None:
        return JSONResponse(status_code=403, content={"status": "error", "reason": "invalid_tag"})

    payload["received_at"] = datetime.now().isoformat(timespec="seconds")
    history.append(payload)
    print("[SERVER]", json.dumps(payload))
    return {"status": "ok", "records": len(history)}


@app.get("/api/status")
def api_status():
    return {
        "model": "SVM",
        "records": len(history),
        "last_record": history[-1] if history else None,
    }


@app.get("/api/history")
def api_history():
    return {
        "model": "SVM",
        "history": history[-100:],
    }


@app.get("/", response_class=HTMLResponse)
def root():
    html = """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>server_hfl_svm</title>
    <style>
        :root {
            --bg: #0b1220;
            --panel: #121a2b;
            --panel-2: #18233a;
            --line: #26324f;
            --text: #eef4ff;
            --muted: #9fb0d1;
            --ok: #32d296;
            --warn: #ffb84d;
        }
        * {
            box-sizing: border-box;
        }
        body {
            margin: 0;
            font-family: "Segoe UI", Tahoma, sans-serif;
            background: radial-gradient(circle at top, #14203a 0%, var(--bg) 55%);
            color: var(--text);
        }
        .wrap {
            max-width: 1100px;
            margin: 0 auto;
            padding: 24px;
        }
        .hero {
            margin-bottom: 20px;
            padding: 22px;
            border: 1px solid var(--line);
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(24,35,58,0.95), rgba(10,17,30,0.96));
            box-shadow: 0 20px 60px rgba(0,0,0,0.25);
        }
        h1 {
            margin: 0 0 8px;
            font-size: 28px;
        }
        .subtitle {
            color: var(--muted);
            margin: 0;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 14px;
            margin-bottom: 20px;
        }
        .card {
            padding: 18px;
            border-radius: 16px;
            border: 1px solid var(--line);
            background: rgba(18, 26, 43, 0.95);
        }
        .label {
            color: var(--muted);
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .value {
            margin-top: 8px;
            font-size: 26px;
            font-weight: 700;
        }
        .status-dot {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 999px;
            background: var(--ok);
            margin-right: 8px;
        }
        .section {
            margin-bottom: 20px;
            padding: 18px;
            border-radius: 16px;
            border: 1px solid var(--line);
            background: rgba(18, 26, 43, 0.94);
        }
        .section h2 {
            margin: 0 0 14px;
            font-size: 18px;
        }
        .empty {
            color: var(--muted);
            padding: 18px;
            border-radius: 12px;
            background: rgba(255,255,255,0.03);
        }
        pre {
            margin: 0;
            overflow: auto;
            padding: 14px;
            border-radius: 12px;
            background: #0a101d;
            border: 1px solid var(--line);
            color: #dce7ff;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid var(--line);
            font-size: 14px;
        }
        th {
            color: var(--muted);
            font-weight: 600;
        }
        .toolbar {
            display: flex;
            gap: 10px;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 10px;
            flex-wrap: wrap;
        }
        .badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.03);
            color: var(--muted);
            font-size: 13px;
        }
    </style>
</head>
<body>
    <div class="wrap">
        <section class="hero">
            <h1>Servidor HFL - SVM</h1>
            <p class="subtitle">Recibe predicciones cifradas con ASCON desde el gateway y las muestra en tiempo real.</p>
        </section>

        <section class="grid">
            <article class="card">
                <div class="label">Servicio</div>
                <div class="value" id="service-name">server_hfl_svm</div>
            </article>
            <article class="card">
                <div class="label">Modelo</div>
                <div class="value" id="model-name">SVM</div>
            </article>
            <article class="card">
                <div class="label">Registros recibidos</div>
                <div class="value" id="records-count">0</div>
            </article>
            <article class="card">
                <div class="label">Estado</div>
                <div class="value"><span class="status-dot"></span>Activo</div>
            </article>
        </section>

        <section class="section">
            <div class="toolbar">
                <h2>Ultimo registro</h2>
                <div class="badge" id="last-updated">Esperando datos...</div>
            </div>
            <pre id="last-record">Aun no han llegado predicciones al servidor.</pre>
        </section>

        <section class="section">
            <h2>Historial reciente</h2>
            <div id="history-container" class="empty">Sin registros todavia. Cuando el gateway envie datos, apareceran aqui.</div>
        </section>
    </div>

    <script>
        function escapeHtml(value) {
            return String(value)
                .replaceAll('&', '&amp;')
                .replaceAll('<', '&lt;')
                .replaceAll('>', '&gt;')
                .replaceAll('"', '&quot;')
                .replaceAll("'", '&#039;');
        }

        function renderTable(records) {
            if (!records.length) {
                return '<div class="empty">Sin registros todavia. Cuando el gateway envie datos, apareceran aqui.</div>';
            }

            const rows = records.slice().reverse().map((record) => {
                return `
                    <tr>
                        <td>${escapeHtml(record.received_at ?? '-')}</td>
                        <td>${escapeHtml(record.client_id ?? '-')}</td>
                        <td>${escapeHtml(record.predicted_label ?? record.attack_type ?? '-')}</td>
                        <td>${typeof record.confidence === 'number' ? record.confidence.toFixed(4) : '-'}</td>
                        <td>${escapeHtml(record.model ?? 'SVM')}</td>
                    </tr>
                `;
            }).join('');

            return `
                <table>
                    <thead>
                        <tr>
                            <th>Recibido</th>
                            <th>Cliente</th>
                            <th>Prediccion</th>
                            <th>Confianza</th>
                            <th>Modelo</th>
                        </tr>
                    </thead>
                    <tbody>${rows}</tbody>
                </table>
            `;
        }

        async function refresh() {
            try {
                const [statusResponse, historyResponse] = await Promise.all([
                    fetch('/api/status'),
                    fetch('/api/history'),
                ]);

                const status = await statusResponse.json();
                const historyPayload = await historyResponse.json();
                const history = historyPayload.history ?? [];

                document.getElementById('records-count').textContent = status.records ?? 0;
                document.getElementById('model-name').textContent = status.model ?? 'SVM';

                const lastRecord = status.last_record ?? null;
                document.getElementById('last-record').textContent = lastRecord
                    ? JSON.stringify(lastRecord, null, 2)
                    : 'Aun no han llegado predicciones al servidor.';

                document.getElementById('last-updated').textContent = lastRecord?.received_at
                    ? `Ultimo mensaje: ${lastRecord.received_at}`
                    : 'Esperando datos...';

                document.getElementById('history-container').innerHTML = renderTable(history);
            } catch (error) {
                document.getElementById('history-container').innerHTML =
                    `<div class="empty">No se pudo consultar el servidor: ${escapeHtml(error.message)}</div>`;
            }
        }

        refresh();
        setInterval(refresh, 2000);
    </script>
</body>
</html>"""
    return HTMLResponse(html)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
