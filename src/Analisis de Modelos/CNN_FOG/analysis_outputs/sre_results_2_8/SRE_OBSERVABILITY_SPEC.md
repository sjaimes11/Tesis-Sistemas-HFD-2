# SRE Observability Spec - hfl_v7-CNN_FOG

Generado a partir de `Results_FOG/#2` a `#8` sin modificar los CSV fuente.

## Executive Summary

- Intentos analizados: `7`
- Rango de intentos: `2` a `8`
- p95 promedio ESP32->RPi decrypt: `4.528 ms`
- p95 promedio RPi->PC decrypt: `38.451 ms`
- Overhead promedio de payload: `1092.931 bytes`
- Accuracy local promedio: `0.9297`
- Loss local promedio: `0.2125`
- Accuracy promedio del pre-FedAvg Fog: `0.9297`
- Loss promedio del pre-FedAvg Fog: `0.2125`
- Accuracy global final promedio: `0.9750`
- Loss global final promedio: `0.0910`
- Completion rate promedio: `0.6667`

## Metric Catalog

| metric_name | exactness | source | aggregation | purpose |
| --- | --- | --- | --- | --- |
| transport.edge_rpi.decrypt.p95_ms | exact_from_csv | ascon_metrics_CNN_FOG_gateway_* | p95(elapsed_ms) WHERE channel='ESP32->RPi' AND operation='decrypt' | SLI de latencia de ingreso al gateway por muestra. |
| transport.rpi_pc.decrypt.p95_ms | exact_from_csv | ascon_metrics_server_* | p95(elapsed_ms) WHERE channel='RPi->PC' AND operation='decrypt' | SLI de latencia de descifrado en servidor para actualizaciones de gateways. |
| transport.pc_rpi.encrypt.p95_ms | exact_from_csv | ascon_metrics_server_* | p95(elapsed_ms) WHERE channel='PC->RPi' AND operation='encrypt' | SLI de despliegue del modelo global desde el servidor. |
| transport.payload.overhead.avg_bytes | exact_from_csv | ascon_metrics_CNN_FOG_gateway_*, ascon_metrics_server_fog_* | avg(overhead_bytes) | Costo promedio en bytes introducido por el envelope de ASCON. |
| transport.payload.expansion.avg_ratio | exact_from_csv | ascon_metrics_CNN_FOG_gateway_*, ascon_metrics_server_fog_* | avg(enc_bytes / pt_bytes) | Factor de expansion promedio del payload transmitido. |
| training.local.accuracy.avg | exact_from_csv | model_metrics_gateway_* | avg(accuracy) WHERE stage='local_train' | Calidad media del entrenamiento local por gateway. |
| training.local.loss.avg | exact_from_csv | model_metrics_gateway_* | avg(loss) WHERE stage='local_train' | Estabilidad promedio del entrenamiento local. |
| fog.preaggregation.accuracy.avg | exact_from_csv | model_metrics_gateway_* | avg(accuracy) WHERE stage='fog_fedavg' | Calidad media del pre-FedAvg realizado por el leader Fog. |
| fog.preaggregation.loss.avg | exact_from_csv | model_metrics_gateway_* | avg(loss) WHERE stage='fog_fedavg' | Estabilidad media del pre-FedAvg realizado por el leader Fog. |
| training.gateway.accuracy.skew.avg | exact_from_csv | model_metrics_gateway_A/B | avg(abs(acc_gateway_A - acc_gateway_B)) por round | Desalineacion entre gateways por ronda. |
| training.gateway.loss.skew.avg | exact_from_csv | model_metrics_gateway_A/B | avg(abs(loss_gateway_A - loss_gateway_B)) por round | Desalineacion de optimizacion entre gateways. |
| round.global.accuracy.last | exact_from_csv | global_weights_history_* | last(accuracy) por intento | Resultado final del modelo global por corrida. |
| round.global.loss.last | exact_from_csv | global_weights_history_* | last(loss) por intento | Punto final de convergencia del modelo global. |
| round.duration.avg_sec | exact_from_csv | global_weights_history_* | avg(diff(time)) por intento | Duracion promedio de ronda federada. |
| round.weight_drift.avg | exact_from_csv | global_weights_history_* | avg(abs(delta(w3_mag))+abs(delta(w4_*))) por round | Movimiento medio de los pesos globales por ronda. |
| reliability.round_completion.rate | exact_from_csv_with_assumption | global_weights_history_* | observed_rounds / expected_rounds_per_attempt | Disponibilidad experimental por corrida. |
| reliability.gateway_participation.rate | exact_from_csv | model_metrics_gateway_* | local_train_rounds / global_rounds_observed | Tasa de participacion efectiva por gateway. |

## Log Catalog

| log_event | status | source | fields | notes |
| --- | --- | --- | --- | --- |
| transport.crypto.gateway | reconstructed_from_csv | ascon_metrics_CNN_FOG_gateway_* | timestamp_dt, attempt_id, gateway_id, channel, operation, elapsed_ms, pt_bytes, enc_bytes, overhead_bytes, client_id, sample_label_name | Incluye trafico ESP32<->RPi y trafico Fog RPi_peer<->RPi_leader sin tocar el CSV original. |
| transport.crypto.server | reconstructed_from_csv | ascon_metrics_server_fog_* | timestamp_dt, attempt_id, channel, operation, elapsed_ms, pt_bytes, enc_bytes, overhead_bytes, round_ref | Representa el intercambio cifrado entre leader Fog y PC. No contiene gateway_id explicito. |
| model.local_train | reconstructed_from_csv | model_metrics_gateway_* | timestamp_dt, attempt_id, gateway_id, fl_round, round_ref, num_samples, accuracy, loss, buffer_target | Representa el fin del entrenamiento local por gateway. |
| model.fog_fedavg | reconstructed_from_csv | model_metrics_gateway_* | timestamp_dt, attempt_id, gateway_id, fl_round, round_ref, num_samples, accuracy, loss, fog_role, peer_count | Representa el pre-FedAvg ejecutado por el leader Fog antes de enviar al servidor. |
| model.global_round | reconstructed_from_csv | global_weights_history_* | timestamp_dt, attempt_id, round_ref, accuracy, loss, w3_mag, w4_normal, w4_brute, w4_scan, fog_samples | Evento canonico de cierre de ronda global agregada desde Fog. |
| fedavg.compute | live_stdout_only | server_hfl_fog.py | trace_id, ts_start, ts_end, round_ref, fog_clusters_received, fedavg_ms | No reconstruible con CSV actuales; recomendado solo a consola JSON del servidor Fog. |
| model.deploy.gateway | live_stdout_only | gateway_hfl_fog.py | trace_id, ts, gateway_id, round_ref, payload_bytes, apply_ms, status | Hoy solo puede estimarse desde los eventos de cifrado del leader Fog y del servidor. |

## Trace Catalog

| trace_name | status | trace_id_pattern | spans | notes |
| --- | --- | --- | --- | --- |
| round_trace | reconstructed_from_csv | attempt:{attempt_id}:round:{round_ref} | gateway_A.local_train, gateway_B.local_train, leader.fog_fedavg, server.decrypt_batch, server.encrypt_global, global_round_commit | Traza exacta a nivel de ronda Fog basada en CSV ya capturados. |
| gateway_round_trace | reconstructed_from_csv | attempt:{attempt_id}:round:{round_ref}:gateway:{gateway_id} | esp32_to_rpi_decrypt_batch, local_train_done, rpi_peer_to_leader_exchange | Traza util para diagnosticar carga, skew y trafico intra-Fog por gateway. |
| sample_trace | live_stdout_only | attempt:{attempt_id}:round:{round_ref}:sample:{client_id}:{sample_seq} | esp32.publish, gateway.decrypt, gateway.buffer, gateway.train_enqueue | No reconstruible con precision usando solo CSV actuales. |
| fedavg_trace | live_stdout_only | attempt:{attempt_id}:round:{round_ref}:fedavg | leader.wait_peer, leader.fog_fedavg, server.wait_fog_clusters, server.fedavg_compute, server.encrypt_global, server.deploy | Requiere logging efimero en tiempo real si se quiere exactitud completa de la jerarquia Fog. |

## Dashboard Panels

| panel_group | panel_name | chart_type | source | metric |
| --- | --- | --- | --- | --- |
| System Health | Round completion rate | gauge/bar | global_weights_history_* | reliability.round_completion.rate |
| System Health | Gateway participation | bar | model_metrics_gateway_* | reliability.gateway_participation.rate |
| Transport | Latency p95 by channel | bar | ascon_metrics_CNN_FOG_* | transport.*.p95_ms |
| Transport | Payload overhead by channel | bar | ascon_metrics_CNN_FOG_* | transport.payload.overhead.avg_bytes |
| Local Training | Local accuracy by gateway | line | model_metrics_gateway_* | training.local.accuracy.avg |
| Local Training | Gateway skew | line | model_metrics_gateway_* | training.gateway.*.skew.avg |
| Global Convergence | Global accuracy and loss | line | global_weights_history_* | round.global.* |
| Global Convergence | Weight drift | line | global_weights_history_* | round.weight_drift.avg |
| Fog Layer | Leader fog_fedavg accuracy and loss | line | model_metrics_gateway_* | fog.preaggregation.* |
| Data Quality | Class mix by gateway | stacked_bar | ascon_metrics_CNN_FOG_gateway_* | data.class_mix.share |

## Scope

- Reconstruible ahora: metricas, logs canonicos y trazas de ronda Fog a partir de CSV historicos.
- Solo live stdout: spans internos de Fog FedAvg, FedAvg en servidor, aplicacion del modelo en gateway y trazas por muestra.