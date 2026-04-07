        # Random Forest — Bundle `hlf_v7`

        Este paquete contiene una versión de despliegue para:
        - ESP32: `esp32/model_weights.h` y `esp32/main_edge_node.cpp`
        - Raspberry Pi: `raspberry/model.pkl`, `raspberry/mqtt_gateway.py`, `raspberry/predict_local.py`

        ## Clases
        - `0` = `normal`
- `1` = `mqtt_bruteforce`
- `2` = `scan_A`

        ## Features
        - `num_pkts`
- `mean_iat`
- `std_iat`
- `min_iat`
- `max_iat`
- `mean_pkt_len`
- `num_bytes`
- `num_psh_flags`
- `num_rst_flags`
- `num_urg_flags`
- `std_pkt_len`
- `min_pkt_len`
- `max_pkt_len`

        ## Tamaño del header ESP32
        - `model_weights.h`: 15805.2 KB

        ## Advertencias
        - El header C excede 4 MB; esta variante es poco realista para ESP32 y está orientada más a Raspberry/validación.

        ## Uso rápido
        - ESP32: usa `main_edge_node_normal_<modelo>.cpp` o `main_edge_node_simulated_<modelo>.cpp` junto a `model_weights.h` y `ascon128.h`.
        - Raspberry Pi: instala `raspberry/requirements.txt` y ejecuta `python gateway_hfl_fog_<modelo>.py`.
