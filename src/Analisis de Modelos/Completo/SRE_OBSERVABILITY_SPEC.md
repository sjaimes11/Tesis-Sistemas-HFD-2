# SRE Observability Spec - hfl_v7 Consolidado

Generado a partir de `Results` y `Results_FOG` de las cuatro carpetas de variante
(RN/CNN x ASCON/PLAIN), sin modificar los CSV fuente. Esta spec extiende el SRE
previo de `Analisis de Modelos/RN/` para cubrir las 8 combinaciones experimentales
y permitir comparacion homogenea.

## Variantes cubiertas

| Variante | Modelo | Seguridad | Topologia | Carpeta |
| --- | --- | --- | --- | --- |
| `RN_ASCON_NoFOG` | RN | ASCON | NoFOG | `hfl_v7-RN\Results` |
| `RN_ASCON_FOG` | RN | ASCON | FOG | `hfl_v7-RN\Results_FOG` |
| `RN_PLAIN_NoFOG` | RN | PLAIN | NoFOG | `hfl_v7-no-ascon-RN\Results` |
| `RN_PLAIN_FOG` | RN | PLAIN | FOG | `hfl_v7-no-ascon-RN\Results_FOG` |
| `CNN_ASCON_NoFOG` | CNN | ASCON | NoFOG | `hfl_v7-CNN\Results` |
| `CNN_ASCON_FOG` | CNN | ASCON | FOG | `hfl_v7-CNN\Results_FOG` |
| `CNN_PLAIN_NoFOG` | CNN | PLAIN | NoFOG | `hfl_v7-no-ascon-CNN\Results` |
| `CNN_PLAIN_FOG` | CNN | PLAIN | FOG | `hfl_v7-no-ascon-CNN\Results_FOG` |

## Executive Summary global

- Variantes con datos: `8`
- Total de intentos analizados: `58`
- Accuracy global final promedio (todas las variantes): `0.9437`
- Loss global final promedio: `0.1383`
- Overhead promedio de payload (bytes): `635.9`
- Duracion promedio de ronda (s): `61.36`

## Executive Summary por variante

| Variante | Intentos | Acc final | Loss final | Round (s) | GW p95 (ms) | Server dec p95 (ms) | Overhead (B) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `CNN_ASCON_FOG` | 7 | 0.9750 | 0.0910 | 58.61 | 4.528 | 37.467 | 1098.4 |
| `CNN_ASCON_NoFOG` | 8 | 0.9604 | 0.1040 | 58.97 | 4.553 | 24.461 | 1049.2 |
| `CNN_PLAIN_FOG` | 3 | 0.9444 | 0.1190 | 54.86 | 0.161 | 0.000 | 0.0 |
| `CNN_PLAIN_NoFOG` | 3 | 0.9389 | 0.1363 | 52.02 | 0.154 | 0.000 | 0.0 |
| `RN_ASCON_FOG` | 5 | 0.9633 | 0.1124 | 71.13 | 4.374 | 50.473 | 1084.0 |
| `RN_ASCON_NoFOG` | 15 | 0.9381 | 0.1545 | 72.91 | 4.654 | 26.690 | 1050.0 |
| `RN_PLAIN_FOG` | 3 | 0.9000 | 0.2199 | 67.14 | 0.159 | 0.000 | 0.0 |
| `RN_PLAIN_NoFOG` | 14 | 0.9298 | 0.1581 | 51.01 | 0.167 | 0.000 | 0.0 |

## Metric Catalog

| Metrica | Exactitud | Fuente | Agregacion | Proposito |
| --- | --- | --- | --- | --- |
| `transport.edge_rpi.decrypt.p95_ms` | exact_from_csv | gateway transport | p95(elapsed_ms) WHERE channel='ESP32->RPi' AND operation='decrypt' | SLI de latencia de ingreso al gateway por muestra. |
| `transport.rpi_pc.decrypt.p95_ms` | exact_from_csv | server transport | p95(elapsed_ms) WHERE channel='RPi->PC' AND operation='decrypt' | SLI de descifrado del agregador. |
| `transport.pc_rpi.encrypt.p95_ms` | exact_from_csv | server transport | p95(elapsed_ms) WHERE channel='PC->RPi' AND operation='encrypt' | SLI de despliegue del modelo global. |
| `transport.payload.overhead.avg_bytes` | exact_from_csv | gateway+server transport | avg(overhead_bytes) | Costo promedio del envelope ASCON. |
| `transport.payload.expansion.avg_ratio` | exact_from_csv | gateway+server transport | avg(enc_bytes / pt_bytes) | Factor de expansion del payload. |
| `training.local.accuracy.avg` | exact_from_csv | model_metrics | avg(accuracy) WHERE stage='local_train' | Calidad media del entrenamiento local. |
| `training.local.loss.avg` | exact_from_csv | model_metrics | avg(loss) WHERE stage='local_train' | Estabilidad media del entrenamiento local. |
| `training.gateway.accuracy.skew.avg` | exact_from_csv | model_metrics A/B | avg(\|acc_A - acc_B\|) por round | Desalineacion entre gateways. |
| `round.global.accuracy.last` | exact_from_csv | global_weights_history | last(accuracy) por intento | Resultado final del modelo global. |
| `round.global.loss.last` | exact_from_csv | global_weights_history | last(loss) por intento | Punto final de convergencia. |
| `round.duration.avg_sec` | exact_from_csv | global_weights_history | avg(diff(time)) por intento | Duracion promedio de ronda. |
| `round.weight_drift.avg` | exact_from_csv | global_weights_history | avg(\|delta(w3,w4*)\|) | Movimiento medio de pesos globales. |
| `reliability.round_completion.rate` | exact_with_assumption | global_weights_history | observed_rounds / expected_rounds | Disponibilidad experimental. |
| `reliability.gateway_participation.rate` | exact_from_csv | model_metrics | local_train_rounds / global_rounds_observed | Tasa efectiva por gateway. |

## Log Catalog

| Evento | Estado | Fuente | Campos clave |
| --- | --- | --- | --- |
| `transport.crypto.gateway` | reconstructed_from_csv | ascon_metrics_*gateway / *plain_metrics_gateway | timestamp, variant, attempt_id, gateway_id, channel, operation, elapsed_ms, pt_bytes, enc_bytes, overhead_bytes, client_id, sample_label_name |
| `transport.crypto.server` | reconstructed_from_csv | ascon_metrics_server / plain_metrics_server | timestamp, variant, attempt_id, channel, operation, elapsed_ms, pt_bytes, enc_bytes, overhead_bytes, round_ref |
| `model.local_train` | reconstructed_from_csv | model_metrics_gateway_* | timestamp, variant, attempt_id, gateway_id, fl_round, num_samples, accuracy, loss, buffer_target |
| `model.global_round` | reconstructed_from_csv | global_weights_history_* | timestamp, variant, attempt_id, round_ref, accuracy, loss, w3_mag, w4_normal, w4_brute, w4_scan |
| `fedavg.compute` | live_stdout_only | server_hfl.py | trace_id, ts_start, ts_end, gateways_received, fedavg_ms |
| `model.deploy.gateway` | live_stdout_only | gateway_hfl.py | trace_id, ts, gateway_id, round_ref, payload_bytes, apply_ms |

## Trace Catalog

| Traza | Estado | Pattern | Spans |
| --- | --- | --- | --- |
| `round_trace` | reconstructed_from_csv | `variant:{v}:attempt:{a}:round:{r}` | gateway_*.local_train, server.decrypt_batch, server.encrypt_global, global_round_commit |
| `gateway_round_trace` | reconstructed_from_csv | `variant:{v}:attempt:{a}:round:{r}:gateway:{gw}` | esp32_to_rpi_decrypt_batch, local_train_done |
| `sample_trace` | live_stdout_only | `variant:{v}:attempt:{a}:round:{r}:sample:{client_id}:{seq}` | esp32.publish, gateway.decrypt, gateway.buffer, gateway.train_enqueue |
| `fedavg_trace` | live_stdout_only | `variant:{v}:attempt:{a}:round:{r}:fedavg` | server.wait_updates, server.fedavg_compute, server.encrypt_global, server.deploy |

## Dashboard panels recomendados

1. **System Health** -- completion rate por variante, gateway participation, last accuracy/loss.
2. **Transport** -- p95 ESP32->RPi/RPi->PC/PC->RPi por variante; overhead bytes y expansion ratio (clave para diferenciar PLAIN vs ASCON).
3. **Local Training** -- accuracy/loss por gateway, skew A vs B, num_samples vs buffer_target.
4. **Global Convergence** -- accuracy/loss por ronda, weight drift por componente.
5. **Data Quality** -- class mix por variante x gateway; deteccion de class imbalance.

## Alcance

- **Reconstruible ahora desde CSV**: metricas, eventos canonicos y trazas a nivel de ronda; con eje `variant` se puede comparar RN vs CNN, ASCON vs PLAIN y NoFOG vs FOG en una misma tabla.
- **Solo live stdout**: spans internos de FedAvg en el servidor, aplicacion del modelo en gateway, trazas por muestra individual y latencia real de red extremo a extremo.

## Diferencias entre variantes para el lector

- En variantes `*_PLAIN_*` los CSV de transport reportan `payload_bytes` (sin tag/nonce).
  Para mantener la semantica unificada se mapean `serialize/deserialize` a `encrypt/decrypt`
  y se replica `payload_bytes` en `pt_bytes` y `enc_bytes` con `overhead_bytes = 0`.
- En variantes FOG el gateway A puede aparecer como `gateway_fog_leader` (no-ascon) o `gateway_A` (ASCON).
  El eje `device_suffix` conserva el valor original del CSV.
- Los datos PLAIN contienen `operation_raw` con la operacion original (`serialize`/`deserialize`) en `canonical_log_events.csv`.