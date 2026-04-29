# SRE Observability Spec - HFL v7 Completo

## Alcance
Este paquete analiza, sin modificar los CSV originales, los resultados de RN, CNN, ASCON, no-ASCON, topologia estandar y topologia Fog.

## Fuentes
- `canonical_log_events.csv`: vista normalizada de transporte, entrenamiento local, agregacion Fog y rondas globales.
- `transport_sli_summary.csv`: SLI de latencia, bytes y overhead por canal, operacion y gateway.
- `local_training_sli_summary.csv`: calidad local por gateway.
- `fog_aggregation_sli_summary.csv`: agregacion Fog cuando existe `stage=fog_fedavg`.
- `global_round_sli_summary.csv`: convergencia global por intento.
- `round_trace_summary.csv`: traza reconstruida por ronda usando `experiment + attempt_id + round_ref`.

## SLIs principales
- `edge_rpi_processing_p95_ms`: p95 de decrypt/deserialize en `ESP32->RPi`.
- `rpi_pc_processing_p95_ms`: p95 de decrypt/deserialize en `RPi->PC`.
- `pc_rpi_processing_p95_ms`: p95 de encrypt/serialize en `PC->RPi`.
- `avg_payload_wire_bytes`: bytes reales enviados por mensaje.
- `avg_overhead_pct`: overhead relativo ASCON respecto a plaintext.
- `round_completion_rate`: rondas observadas / 30.
- `gateway_sync_skew_sec`: desalineacion temporal entre gateways por ronda.
- `local_accuracy_skew`: diferencia entre mejor y peor gateway en la misma ronda.

## Eventos canonicos
- `transport.crypto.gateway`: operacion ASCON en gateway.
- `transport.crypto.server`: operacion ASCON en servidor.
- `transport.plain.gateway`: serializacion/deserializacion sin ASCON en gateway.
- `transport.plain.server`: serializacion/deserializacion sin ASCON en servidor.
- `model.local_train`: entrenamiento local terminado.
- `model.fog_fedavg`: agregacion Fog terminada.
- `model.global_round`: ronda global registrada.

## Trazas
La traza reconstruida por ronda se guarda en `round_trace_summary.csv`.
Join key: `experiment, attempt_id, round_ref`.

Spans reconstruibles:
- `edge_rpi_transport`
- `local_train`
- `fog_fedavg`
- `rpi_pc_transport`
- `pc_rpi_transport`
- `global_state`

## Dashboard tipo Grafana/SRE
Paneles recomendados:
- Salud del sistema: rondas completadas, ultimo accuracy/loss, duracion p95.
- Transporte: p95 por canal, bytes enviados, overhead ASCON.
- Entrenamiento local: accuracy/loss por gateway y skew entre gateways.
- Fog: eventos `fog_fedavg`, peers y metrica agregada local.
- Convergencia global: accuracy/loss global, `w3_mag`, `w4_normal`, `w4_brute`, `w4_scan`.
- Calidad de datos: mezcla de clases por gateway.
- NIST/ASCON: operaciones ASCON, p95 de cifrado/descifrado, overhead.

## NIST
NIST selecciono Ascon como familia de criptografia ligera para estandarizacion. Este analisis demuestra que el proyecto usa ASCON en rutas IoT/Fog/Cloud y mide su costo operacional.

Importante:
- `>=1000 operaciones`, `p95 <= 50 ms` y `overhead <= 50%` son criterios operativos del proyecto, no umbrales oficiales NIST.
- NIST SP 800-22 es una suite de pruebas estadisticas para generadores aleatorios/pseudoaleatorios, no una guia directa para t-test/Wilcoxon de metricas FL.

## Limitaciones
- No hay latencia end-to-end exacta por muestra sin `trace_id` compartido.
- Algunas filas de servidor no incluyen `gateway_id`, por lo que se tratan como agregadas.
- Las rondas dentro de una ejecucion no son muestras independientes.
