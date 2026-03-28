"""
=============================================================================
 benchmark_ascon.py — Medición de Overhead ASCON-128 para Tesis
=============================================================================
 Pregunta de investigación:
   ¿Cuál es el overhead del cifrado ASCON en la comunicación de modelos federados?

 Métricas:
   1. Overhead computacional: tiempo de cifrado/descifrado (ms)
   2. Overhead de comunicación: tamaño plaintext vs cifrado (bytes y %)
   3. Overhead por canal: ESP32->RPi, RPi->PC, PC->RPi->ESP32
   4. Throughput: mensajes/segundo posibles

 Ejecutar: python benchmark_ascon.py
 
 Genera tablas LaTeX para incluir directamente en el documento de tesis.
=============================================================================
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import time
import base64
import statistics
import numpy as np
from ascon128 import encrypt as ascon_encrypt, decrypt as ascon_decrypt, generate_nonce

ASCON_KEY = bytes([0xA1, 0xB2, 0xC3, 0xD4, 0xE5, 0xF6, 0x07, 0x18,
                   0x29, 0x3A, 0x4B, 0x5C, 0x6D, 0x7E, 0x8F, 0x90])

ITERATIONS = 1000

# ====================== PAYLOADS REALISTAS ======================

def generate_features_payload():
    """Payload típico ESP32 -> RPi (features de un flujo)"""
    return json.dumps({
        "client_id": "esp32_edge_normal_1",
        "features": [14.0, 0.152, 0.023, 0.12, 0.18, 90.0, 1260.0, 14.0, 0.0, 0.0, 5.2, 85.0, 95.0]
    }).encode('utf-8')

def generate_weights_payload():
    """Payload típico RPi -> PC (pesos entrenados localmente)"""
    W3 = np.random.randn(16, 8).tolist()
    b3 = np.random.randn(8).tolist()
    W4 = np.random.randn(8, 3).tolist()
    b4 = np.random.randn(3).tolist()
    return json.dumps({
        "gateway_id": "gateway_A",
        "num_samples": 25,
        "round": 1,
        "accuracy": 0.92,
        "loss": 0.18,
        "W3": W3, "b3": b3,
        "W4": W4, "b4": b4
    }).encode('utf-8')

def generate_global_model_payload():
    """Payload típico PC -> RPi -> ESP32 (modelo global)"""
    W3 = np.random.randn(16, 8).tolist()
    b3 = np.random.randn(8).tolist()
    W4 = np.random.randn(8, 3).tolist()
    b4 = np.random.randn(3).tolist()
    return json.dumps({
        "round": 1,
        "W3": W3, "b3": b3,
        "W4": W4, "b4": b4
    }).encode('utf-8')


# ====================== BENCHMARK ======================

def benchmark_channel(name, plaintext_bytes, iterations=ITERATIONS):
    """Mide overhead completo para un canal de comunicación"""
    pt_size = len(plaintext_bytes)
    
    # Medir cifrado
    enc_times = []
    for i in range(iterations):
        nonce = generate_nonce(int(time.time() * 1000) & 0xFFFFFFFF, i)
        t0 = time.perf_counter()
        ct, tag = ascon_encrypt(plaintext_bytes, ASCON_KEY, nonce)
        t1 = time.perf_counter()
        enc_times.append((t1 - t0) * 1000)
    
    # Medir descifrado
    nonce = generate_nonce(12345, 99)
    ct, tag = ascon_encrypt(plaintext_bytes, ASCON_KEY, nonce)
    
    dec_times = []
    for i in range(iterations):
        t0 = time.perf_counter()
        pt = ascon_decrypt(ct, ASCON_KEY, nonce, tag)
        t1 = time.perf_counter()
        dec_times.append((t1 - t0) * 1000)
        assert pt == plaintext_bytes
    
    # Overhead de tamaño (formato JSON con base64)
    envelope = json.dumps({
        "ct": base64.b64encode(ct).decode('ascii'),
        "tag": base64.b64encode(tag).decode('ascii'),
        "nonce": base64.b64encode(nonce).decode('ascii')
    }).encode('utf-8')
    enc_size = len(envelope)
    
    return {
        "name": name,
        "pt_size": pt_size,
        "ct_raw_size": len(ct) + 16 + 16,  # ct + tag + nonce
        "enc_envelope_size": enc_size,
        "size_overhead_bytes": enc_size - pt_size,
        "size_overhead_pct": ((enc_size - pt_size) / pt_size) * 100,
        "enc_mean_ms": statistics.mean(enc_times),
        "enc_median_ms": statistics.median(enc_times),
        "enc_std_ms": statistics.stdev(enc_times),
        "enc_min_ms": min(enc_times),
        "enc_max_ms": max(enc_times),
        "dec_mean_ms": statistics.mean(dec_times),
        "dec_median_ms": statistics.median(dec_times),
        "dec_std_ms": statistics.stdev(dec_times),
        "dec_min_ms": min(dec_times),
        "dec_max_ms": max(dec_times),
        "throughput_msgs_sec": 1000.0 / statistics.mean(enc_times),
    }


def print_results(results):
    print("\n" + "=" * 80)
    print(" RESULTADOS: Overhead de ASCON-128 en Comunicación Federada")
    print(f" Iteraciones por medición: {ITERATIONS}")
    print(f" Plataforma: Python (equivalente a Raspberry Pi / PC)")
    print("=" * 80)
    
    # Tabla 1: Overhead de Tamaño
    print("\n" + "─" * 80)
    print(" TABLA 1: Overhead de Comunicación (Tamaño de Mensajes)")
    print("─" * 80)
    print(f"{'Canal':<25} {'Plaintext':>10} {'Cifrado':>10} {'Overhead':>10} {'%':>8}")
    print(f"{'':.<25} {'(bytes)':>10} {'(bytes)':>10} {'(bytes)':>10} {'':>8}")
    print("─" * 80)
    for r in results:
        print(f"{r['name']:<25} {r['pt_size']:>10} {r['enc_envelope_size']:>10} "
              f"{r['size_overhead_bytes']:>+10} {r['size_overhead_pct']:>7.1f}%")
    
    # Tabla 2: Overhead Computacional
    print("\n" + "─" * 80)
    print(" TABLA 2: Overhead Computacional (Tiempo de Cifrado/Descifrado)")
    print("─" * 80)
    print(f"{'Canal':<25} {'Enc (ms)':>10} {'Dec (ms)':>10} {'Total (ms)':>11} {'Msgs/s':>10}")
    print("─" * 80)
    for r in results:
        total = r['enc_mean_ms'] + r['dec_mean_ms']
        print(f"{r['name']:<25} {r['enc_mean_ms']:>10.3f} {r['dec_mean_ms']:>10.3f} "
              f"{total:>11.3f} {r['throughput_msgs_sec']:>10.0f}")
    
    # Tabla 3: Distribución estadística
    print("\n" + "─" * 80)
    print(" TABLA 3: Distribución Estadística del Cifrado (ms)")
    print("─" * 80)
    print(f"{'Canal':<25} {'Media':>8} {'Mediana':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("─" * 80)
    for r in results:
        print(f"{r['name']:<25} {r['enc_mean_ms']:>8.3f} {r['enc_median_ms']:>8.3f} "
              f"{r['enc_std_ms']:>8.4f} {r['enc_min_ms']:>8.3f} {r['enc_max_ms']:>8.3f}")
    
    # Overhead total por ronda federada
    print("\n" + "─" * 80)
    print(" TABLA 4: Overhead Total por Ronda de Aprendizaje Federado")
    print("─" * 80)
    
    feat = next(r for r in results if "features" in r['name'].lower())
    wt = next(r for r in results if "pesos" in r['name'].lower())
    gm = next(r for r in results if "global" in r['name'].lower())
    
    n_features_per_round = 25  # SAMPLES_PER_UPDATE
    n_gateways = 2
    
    # Por ronda: 25 features cifradas + 25 descifradas + 1 peso cifrado + 1 descifrado + 1 global cifrado + 1 descifrado
    enc_features = n_features_per_round * (feat['enc_mean_ms'] + feat['dec_mean_ms'])
    enc_weights = n_gateways * (wt['enc_mean_ms'] + wt['dec_mean_ms'])
    enc_global = n_gateways * (gm['enc_mean_ms'] + gm['dec_mean_ms'])
    total_crypto = enc_features + enc_weights + enc_global
    
    size_features = n_features_per_round * feat['size_overhead_bytes']
    size_weights = n_gateways * wt['size_overhead_bytes']
    size_global = n_gateways * gm['size_overhead_bytes']
    total_size = size_features + size_weights + size_global
    
    print(f"  {n_features_per_round}x Features (ESP32->RPi):    {enc_features:>8.1f} ms  |  +{size_features:>6} bytes")
    print(f"  {n_gateways}x Pesos (RPi->PC):           {enc_weights:>8.1f} ms  |  +{size_weights:>6} bytes")
    print(f"  {n_gateways}x Global (PC->RPi->ESP32):    {enc_global:>8.1f} ms  |  +{size_global:>6} bytes")
    print(f"  {'─'*55}")
    print(f"  TOTAL por ronda:             {total_crypto:>8.1f} ms  |  +{total_size:>6} bytes")
    
    typical_train_time_ms = 2000  # ~2 segundos de entrenamiento Keras en RPi
    total_round_ms = typical_train_time_ms + total_crypto
    crypto_pct = (total_crypto / total_round_ms) * 100
    
    print(f"\n  Tiempo entrenamiento local (referencia):  ~{typical_train_time_ms} ms")
    print(f"  Overhead criptográfico vs entrenamiento:   {crypto_pct:.2f}%")
    
    print("\n" + "─" * 80)
    print(" CONCLUSIÓN")
    print("─" * 80)
    if crypto_pct < 5:
        print(f"  El overhead de ASCON-128 es NEGLIGIBLE ({crypto_pct:.2f}% del tiempo total)")
        print("  de una ronda federada, lo que confirma su idoneidad para IoT")
        print("  con restricciones de recursos.")
    elif crypto_pct < 20:
        print(f"  El overhead de ASCON-128 es BAJO ({crypto_pct:.2f}% del tiempo total)")
        print("  de una ronda federada, aceptable para aplicaciones IoT.")
    else:
        print(f"  El overhead de ASCON-128 es SIGNIFICATIVO ({crypto_pct:.2f}% del tiempo total)")
    
    return results


def generate_latex_tables(results):
    """Genera tablas LaTeX para copiar directamente en la tesis"""
    feat = next(r for r in results if "features" in r['name'].lower())
    wt = next(r for r in results if "pesos" in r['name'].lower())
    gm = next(r for r in results if "global" in r['name'].lower())
    
    print("\n\n" + "=" * 80)
    print(" TABLAS LATEX (copiar al documento de tesis)")
    print("=" * 80)
    
    print(r"""
% Tabla: Overhead de tamaño
\begin{table}[h]
\centering
\caption{Overhead de comunicación por cifrado ASCON-128}
\label{tab:ascon-size-overhead}
\begin{tabular}{lcccr}
\hline
\textbf{Canal} & \textbf{Plaintext} & \textbf{Cifrado} & \textbf{Overhead} & \textbf{\%} \\
 & (bytes) & (bytes) & (bytes) & \\
\hline""")
    for r in results:
        name_latex = r['name'].replace('→', r'$\rightarrow$')
        print(f"{name_latex} & {r['pt_size']} & {r['enc_envelope_size']} & "
              f"+{r['size_overhead_bytes']} & {r['size_overhead_pct']:.1f}\\% \\\\")
    print(r"""\hline
\end{tabular}
\end{table}""")

    print(r"""
% Tabla: Overhead computacional
\begin{table}[h]
\centering
\caption{Tiempo de cifrado/descifrado ASCON-128 por canal}
\label{tab:ascon-time-overhead}
\begin{tabular}{lcccc}
\hline
\textbf{Canal} & \textbf{Cifrado} & \textbf{Descifrado} & \textbf{Total} & \textbf{Throughput} \\
 & (ms) & (ms) & (ms) & (msg/s) \\
\hline""")
    for r in results:
        name_latex = r['name'].replace('→', r'$\rightarrow$')
        total = r['enc_mean_ms'] + r['dec_mean_ms']
        print(f"{name_latex} & {r['enc_mean_ms']:.3f} & {r['dec_mean_ms']:.3f} & "
              f"{total:.3f} & {r['throughput_msgs_sec']:.0f} \\\\")
    print(r"""\hline
\end{tabular}
\end{table}""")


# ====================== MAIN ======================
if __name__ == "__main__":
    print("=" * 80)
    print(" BENCHMARK ASCON-128 PARA APRENDIZAJE FEDERADO")
    print(f" Midiendo {ITERATIONS} iteraciones por canal...")
    print("=" * 80)
    
    channels = [
        ("Features (ESP32→RPi)", generate_features_payload()),
        ("Pesos (RPi→PC)", generate_weights_payload()),
        ("Modelo Global (PC→RPi)", generate_global_model_payload()),
    ]
    
    results = []
    for name, payload in channels:
        print(f"\n  Benchmarking: {name} ({len(payload)} bytes)...")
        r = benchmark_channel(name, payload)
        results.append(r)
    
    print_results(results)
    generate_latex_tables(results)
    
    print("\n\nBenchmark completado.")
