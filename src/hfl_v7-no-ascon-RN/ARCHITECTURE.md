# HFL v7-no-ascon — Hierarchical Federated Learning for IoT IDS

Baseline derivado de `hfl_v7-RN` para medir latencia y overhead sin cifrado.

- MQTT y HTTP usan JSON plano.
- `plain_metrics.py` reemplaza al recolector ASCON para medir serialización y parseo.
- `results_model/` conserva las métricas de accuracy/loss por gateway.

## Architecture Overview

```
                          ┌─────────────────────────┐
                          │     PC (Cloud Server)    │
                          │   server_hfl.py          │
                          │   server_hfl_fog.py      │
                          │   FedAvg + Dashboard     │
                          └────────┬───────┬─────────┘
                         HTTP+ASCON│       │HTTP+ASCON
                     ┌─────────────┘       └──────────────┐
                     ▼                                     ▼
           ┌──────────────────┐  fog MQTT+ASCON  ┌──────────────────┐
           │  Raspberry Pi A  │◄────────────────►│  Raspberry Pi B  │
           │  (Fog Leader)    │                  │  (Fog Peer)      │
           │  gateway_hfl.py  │                  │  gateway_hfl.py  │
           │  gateway_hfl_    │                  │  gateway_hfl_    │
           │    fog.py        │                  │    fog.py        │
           └───────┬──────────┘                  └───────┬──────────┘
          MQTT+ASCON│                            MQTT+ASCON│
           ┌───────┴───────┐                     ┌───────┴───────┐
           ▼               ▼                     ▼               ▼
     ┌──────────┐   ┌──────────┐           ┌──────────┐   ┌──────────┐
     │ ESP32-S3 │   │ ESP32-S3 │           │ ESP32-S3 │   │ ESP32-S3 │
     │ (Normal) │   │(Simulated│           │ (Normal) │   │(Simulated│
     │          │   │  Attacks)│           │          │   │  Attacks)│
     └──────────┘   └──────────┘           └──────────┘   └──────────┘
```

There are **two deployment modes**:

| Mode | Files | Description |
|------|-------|-------------|
| **Standard (2-tier)** | `gateway_hfl.py` + `server_hfl.py` | Each RPi sends directly to PC. PC aggregates from K gateways. |
| **Fog (3-tier)** | `gateway_hfl_fog.py` + `server_hfl_fog.py` | RPis pre-aggregate between themselves, then the fog leader sends to PC. |

Both modes use ASCON-128 encryption on **every** communication channel.

---

## Files — Role in the Architecture

### Edge Layer (ESP32-S3 Microcontrollers)

| File | Device | Role |
|------|--------|------|
| `main_edge_node_normal.cpp` | ESP32-S3 | Generates **100% normal** MQTT traffic. Extracts 13 features per flow, runs TinyML inference locally, sends features to RPi via MQTT (`fl/features`), receives global model updates via MQTT (`fl/global_model`). All MQTT payloads encrypted with ASCON-128. |
| `main_edge_node_simulated.cpp` | ESP32-S3 | Generates **mixed traffic** (40% normal, 30% bruteforce, 30% scan_A) with statistical distributions matching real datasets. Same feature extraction, inference, and communication as the normal node. |
| `main_edge_node.cpp` | ESP32-S3 | **Legacy/alternative** node that runs a local MQTT broker (`sMQTTBroker`) to capture traffic from smaller IoT sensors. Does not use ASCON. Kept for reference. |
| `ascon128.h` | ESP32-S3 | C++ implementation of ASCON-128 authenticated encryption/decryption. Included by both `main_edge_node_normal.cpp` and `main_edge_node_simulated.cpp`. Uses the ESP32's `mbedtls` Base64 library for encoding encrypted payloads into JSON. |
| `model_weights.h` | ESP32-S3 | Auto-generated header containing the MLP model weights (`13→32→16→8→3`), biases, and StandardScaler parameters (`scaler_mean`, `scaler_std`). Used by all ESP32 nodes for on-device inference. |

### Fog Layer (Raspberry Pi 4)

| File | Device | Role |
|------|--------|------|
| `gateway_hfl.py` | Raspberry Pi 4 | **Standard mode gateway.** Acts as MQTT broker (Mosquitto) for its ESP32 cluster. Receives encrypted features from ESP32s, decrypts with ASCON, labels samples using a heuristic function, fills a training buffer (40 samples), trains the local Keras MLP (5 epochs), encrypts and sends weights to PC via HTTP, receives encrypted global model from PC, and distributes it to ESP32s. Handles the full bidirectional flow. |
| `gateway_hfl_fog.py` | Raspberry Pi 4 | **Fog mode gateway.** Single file that runs as either **leader** or **peer** (configured via `FOG_ROLE` env var). Adds inter-gateway MQTT communication on topics `fog/weights` and `fog/global_model`. The leader collects local weights + peer weights, performs Fog FedAvg, then sends the pre-aggregated result to PC. The leader also distributes the global model back to peers. The peer sends its weights to the leader and receives the global model from the leader. All fog communication is encrypted with ASCON-128. |

### Cloud Layer (PC Server)

| File | Device | Role |
|------|--------|------|
| `server_hfl.py` | PC | **Standard mode server.** FastAPI application that receives encrypted weights from K=2 individual gateways, decrypts with ASCON, applies sample-weighted FedAvg when all gateways report, encrypts and distributes the global model back to each gateway. Includes a real-time analytics dashboard (Chart.js) at `http://localhost:8001/` showing accuracy, loss, and weight magnitudes per round. |
| `server_hfl_fog.py` | PC | **Fog mode server.** Receives pre-aggregated weights from fog cluster leaders (endpoint `/aggregate-from-fog`). `MIN_UPDATES_PER_ROUND=1` for a single fog cluster. If multiple fog clusters exist, performs global FedAvg across clusters. Dashboard updated with fog-specific information. |

### Cryptography & Metrics

| File | Device | Role |
|------|--------|------|
| `ascon128.py` | RPi + PC | Python implementation of ASCON-128 AEAD (NIST LWC standard). Provides `encrypt(plaintext, key, nonce) → (ciphertext, tag)` and `decrypt(ciphertext, key, nonce, tag) → plaintext | None`. Also provides `generate_nonce(timestamp_ms, counter)` with 32-bit truncation to prevent overflow. Used by all Python scripts (gateway and server). |
| `ascon128.h` | ESP32-S3 | C++ implementation (see Edge Layer above). Interoperable with `ascon128.py` — messages encrypted in C++ can be decrypted in Python and vice versa. |
| `ascon_metrics.py` | RPi + PC | Real-time metrics collector. The `AsconMetrics` class records every encryption/decryption operation with timestamp, channel, time (ms), plaintext size, encrypted size, and overhead. Prints periodic summaries to console and exports to CSV (`ascon_metrics_<device>.csv`) on exit. |

### Testing & Benchmarking

| File | Device | Role |
|------|--------|------|
| `test_ascon.py` | Any | Unit tests for `ascon128.py`. Verifies encrypt/decrypt roundtrips on short messages, long messages, and JSON payloads. Confirms tag verification rejects tampered data. |
| `benchmark_ascon.py` | Any | Benchmarks ASCON-128 performance across different payload sizes (features, weights, full model). Measures encryption/decryption time, throughput, and size overhead. Generates LaTeX tables for the thesis document. |

---

## Communication Channels

All channels use JSON payloads. With ASCON enabled, the JSON payload is encrypted and wrapped in an envelope:

```json
{
  "ct":    "<Base64(ciphertext)>",
  "tag":   "<Base64(16-byte auth tag)>",
  "nonce": "<Base64(16-byte nonce)>"
}
```

### Standard Mode (gateway_hfl.py + server_hfl.py)

| # | Direction | Protocol | Topic / Endpoint | Content | ASCON |
|---|-----------|----------|------------------|---------|-------|
| 1 | ESP32 → RPi | MQTT | `fl/features` | `{client_id, features[13]}` | Yes |
| 2 | RPi → PC | HTTP POST | `/aggregate-from-gateway` | `{gateway_id, W3, b3, W4, b4, accuracy, loss, num_samples}` | Yes |
| 3 | PC → RPi | HTTP POST | `/deploy-model` | `{W3, b3, W4, b4, round}` | Yes |
| 4 | RPi → ESP32 | MQTT | `fl/global_model` | `{W3, b3, W4, b4, round}` | Yes |

### Fog Mode (gateway_hfl_fog.py + server_hfl_fog.py)

All 4 channels above, **plus**:

| # | Direction | Protocol | Topic / Endpoint | Content | ASCON |
|---|-----------|----------|------------------|---------|-------|
| 5 | RPi_peer → RPi_leader | MQTT | `fog/weights` | `{gateway_id, W3, b3, W4, b4, accuracy, loss, num_samples}` | Yes |
| 6 | RPi_leader → RPi_peer | MQTT | `fog/global_model` | `{W3, b3, W4, b4, round}` | Yes |

In fog mode, channel 2 uses endpoint `/aggregate-from-fog` and the payload contains fog-aggregated weights.

---

## Model Architecture

```
Input (13 features) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(8, ReLU) → Dense(3, Softmax)
                       W1, b1            W2, b2            W3, b3           W4, b4
                       448 params         528 params        136 params       27 params
                                                           ╰── shared via FL ──╯
```

- **Total parameters:** 1,139 (~4.5 KB in float32)
- **FL parameters exchanged:** 163 (W3 + b3 + W4 + b4 = ~652 bytes before JSON)
- **Classes:** `normal` (0), `mqtt_bruteforce` (1), `scan_A` (2)

---

## Deployment

### Standard Mode

```bash
# PC (Cloud Server)
cd src/hfl_v7
python server_hfl.py
# Dashboard: http://localhost:8001/

# Raspberry Pi A
cd src/hfl_v7
python gateway_hfl.py
# Starts MQTT broker + HTTP server on :5000

# Raspberry Pi B (optional second gateway)
# Edit GATEWAY_ID and IP_PC in gateway_hfl.py
python gateway_hfl.py

# ESP32-S3 (Normal Node)
# Flash main_edge_node_normal.cpp via PlatformIO/Arduino IDE

# ESP32-S3 (Simulated Attack Node)
# Flash main_edge_node_simulated.cpp via PlatformIO/Arduino IDE
```

### Fog Mode

```bash
# PC (Cloud Server)
cd src/hfl_v7
python server_hfl_fog.py

# Raspberry Pi A (Fog Leader)
FOG_ROLE=leader GATEWAY_ID=gateway_fog_A python gateway_hfl_fog.py
# Runs: local MQTT broker + fog MQTT listener + HTTP server on :5000

# Raspberry Pi B (Fog Peer)
FOG_ROLE=peer GATEWAY_ID=gateway_fog_B python gateway_hfl_fog.py
# Runs: local MQTT broker + connects to leader's MQTT for fog topics

# ESP32 nodes: same as standard mode (no changes needed)
```

### Network Configuration

Edit these IPs in the files before deploying:

| Variable | File(s) | Description |
|----------|---------|-------------|
| `IP_PC` | `gateway_hfl.py`, `gateway_hfl_fog.py` | PC server IP |
| `GATEWAYS` / `FOG_LEADERS` | `server_hfl.py`, `server_hfl_fog.py` | Gateway/leader IPs |
| `FOG_LEADER_IP` | `gateway_hfl_fog.py` | Leader RPi IP (peers use this) |
| `STA_SSID`, `STA_PASS` | `.cpp` files | WiFi credentials |
| `MQTT_HOST` | `.cpp` files | RPi broker IP |

---

## Branches

| Branch | Description |
|--------|-------------|
| `main` | Full system with ASCON-128 enabled on all channels |
| `hfl_v7-no-ascon` | Identical system with ASCON removed — plain JSON on all channels. Used as baseline for measuring encryption overhead. |
