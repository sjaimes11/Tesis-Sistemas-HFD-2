        # Decision Tree — Bundle `hlf_v7`

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
        - `model_weights.h`: 61.5 KB

        ## Advertencias
        - Sin advertencias adicionales.

        ## Uso rápido
        - ESP32: copia `esp32/model_weights.h` y `esp32/main_edge_node.cpp` a tu sketch/firmware.
        - Raspberry Pi: instala `raspberry/requirements.txt` y ejecuta `python raspberry/mqtt_gateway.py`.
