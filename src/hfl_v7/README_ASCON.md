# HFL v7 con ASCON-128 - Guía de Deployment

## Arquitectura

```
ESP32 (x2) ──[ASCON/MQTT]──> Raspberry Pi 4 ──[ASCON/HTTP]──> PC
```

**Dispositivos:**
- 2x ESP32-S3: `main_edge_node_normal.cpp` + `main_edge_node_simulated.cpp`
- 1x Raspberry Pi 4: `gateway_hfl.py`
- 1x PC: `server_hfl.py`

**Seguridad:** Todos los canales cifrados con ASCON-128 (NIST LWC 2023)

---

## 1. Preparación de archivos

### ESP32 requiere:
- `ascon128.h` (copiar a la carpeta del proyecto PlatformIO/Arduino)
- `model_weights.h` (ya existente)
- `main_edge_node_normal.cpp` O `main_edge_node_simulated.cpp`

### Raspberry Pi requiere:
- `gateway_hfl.py`
- `ascon128.py`
- `ids_3class.keras` (modelo base)
- Mosquitto: `sudo apt install mosquitto mosquitto-clients`
- Python: `pip install paho-mqtt numpy requests tensorflow`

### PC requiere:
- `server_hfl.py`
- `ascon128.py`
- Python: `pip install fastapi uvicorn numpy requests`

---

## 2. Configuración de IPs

**Editar en cada archivo:**

`main_edge_node_*.cpp` (línea 30):
```cpp
const char* GATEWAY_MQTT_SERVER = "192.168.X.X"; // IP de la Raspberry Pi
```

`gateway_hfl.py` (línea 29):
```python
IP_PC = "192.168.X.X"  # IP del PC
```

`server_hfl.py` (línea 60-62):
```python
GATEWAYS = [
    "http://192.168.X.X:5000",  # IP Raspberry Pi
]
```

---

## 3. Deployment

### Paso 1: PC (Servidor Central)
```bash
cd /ruta/a/hfl_v7
python server_hfl.py
```
Dashboard: http://localhost:8001/

### Paso 2: Raspberry Pi (Gateway)
```bash
sudo systemctl start mosquitto
cd /ruta/a/hfl_v7
python gateway_hfl.py
```

### Paso 3: ESP32 (Nodos Edge)
1. Abrir PlatformIO o Arduino IDE
2. Flashear `main_edge_node_normal.cpp` en ESP32 #1
3. Flashear `main_edge_node_simulated.cpp` en ESP32 #2

---

## 4. Verificación

Logs esperados:

**ESP32:**
```
[FL] ASCON: Descifrado exitoso y autenticado.
[FL] Pesos actualizados en memoria.
```

**Raspberry Pi:**
```
[DATASET] esp32_edge_normal_1 -> normal | Buffer: 12/25 | Nodos: 2
[ASCON] Pesos cifrados. Enviando al servidor PC...
```

**PC:**
```
[SERVIDOR] Pesos recibidos (ASCON OK) del Gateway 'gateway_A' con 25 muestras
FEDAVG GLOBAL - Ronda 1 completada
```

---

## 5. Clave ASCON

**Clave pre-compartida** (idéntica en los 3 componentes):
```
A1 B2 C3 D4 E5 F6 07 18 29 3A 4B 5C 6D 7E 8F 90
```

Para producción, generar con:
```python
import secrets
key = secrets.token_bytes(16)
print(key.hex())
```

---

## Troubleshooting

**ESP32 no descifra:**
- Verificar que `ascon128.h` esté en la carpeta del proyecto
- Verificar que `ASCON_KEY` sea idéntica en los 3 lugares

**Gateway rechaza mensajes:**
- Revisar logs: "ASCON: Tag inválido" indica clave incorrecta o mensaje corrupto

**PC no recibe:**
- Verificar firewall: `sudo ufw allow 8001`
- Verificar conectividad: `curl http://IP_PC:8001/`
