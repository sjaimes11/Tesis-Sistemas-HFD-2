import base64
import json
import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from ascon128 import decrypt as ascon_decrypt
from ascon_metrics import AsconMetrics


BASE_DIR = Path(__file__).resolve().parent
ASCON_KEY = bytes([0xA1, 0xB2, 0xC3, 0xD4, 0xE5, 0xF6, 0x07, 0x18,
                   0x29, 0x3A, 0x4B, 0x5C, 0x6D, 0x7E, 0x8F, 0x90])

metrics = AsconMetrics("server_decision_tree")
app = FastAPI(title="server_hfl_decision_tree")
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
        return {"status": "error", "reason": "invalid_tag"}

    payload["received_at"] = datetime.now().isoformat(timespec="seconds")
    history.append(payload)
    print("[SERVER]", json.dumps(payload))
    return {"status": "ok", "records": len(history)}


@app.get("/api/status")
def api_status():
    return {
        "model": "Decision Tree",
        "records": len(history),
        "last_record": history[-1] if history else None,
    }


@app.get("/")
def root():
    return {
        "service": "server_hfl_decision_tree",
        "model": "Decision Tree",
        "records": len(history),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
