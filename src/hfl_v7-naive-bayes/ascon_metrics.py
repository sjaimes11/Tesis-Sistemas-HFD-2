"""
=============================================================================
 ascon_metrics.py — Recolector de métricas ASCON en tiempo real
=============================================================================
 Se importa desde gateway_hfl.py y server_hfl.py para medir overhead
 criptográfico en producción.

 Genera: ascon_metrics_<device>.csv con una fila por operación
 Muestra resumen periódico en consola
 Al finalizar (Ctrl+C), imprime tablas resumen y exporta CSV
=============================================================================
"""
import time
import csv
import os
import atexit
from datetime import datetime

class AsconMetrics:
    def __init__(self, device_name="unknown"):
        self.device_name = device_name
        self.records = []
        self.start_time = time.time()
        self.csv_path = f"ascon_metrics_{device_name}.csv"
        atexit.register(self.export_summary)
    
    def record(self, channel, operation, pt_size, enc_size, elapsed_ms, fl_round):
        """
        channel:   "ESP32->RPi", "RPi->PC", "PC->RPi", "RPi->ESP32"
        operation: "encrypt" o "decrypt"
        pt_size:   tamaño del plaintext (bytes)
        enc_size:  tamaño del envelope cifrado (bytes)
        elapsed_ms: tiempo de la operación ASCON (ms)
        fl_round:  ronda federada actual
        """
        entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "uptime_s": round(time.time() - self.start_time, 1),
            "channel": channel,
            "operation": operation,
            "pt_bytes": pt_size,
            "enc_bytes": enc_size,
            "overhead_bytes": enc_size - pt_size,
            "elapsed_ms": round(elapsed_ms, 3),
            "fl_round": fl_round,
        }
        self.records.append(entry)
        
        tag = "ENC" if operation == "encrypt" else "DEC"
        print(f"  [ASCON {tag}] {channel}: {elapsed_ms:.3f}ms | {pt_size}B -> {enc_size}B (+{enc_size - pt_size}B)")
    
    def get_summary_by_channel(self):
        from collections import defaultdict
        channels = defaultdict(lambda: {"enc_times": [], "dec_times": [], "overheads": [], "pt_sizes": []})
        
        for r in self.records:
            ch = channels[r["channel"]]
            if r["operation"] == "encrypt":
                ch["enc_times"].append(r["elapsed_ms"])
            else:
                ch["dec_times"].append(r["elapsed_ms"])
            ch["overheads"].append(r["overhead_bytes"])
            ch["pt_sizes"].append(r["pt_bytes"])
        
        return dict(channels)
    
    def print_live_summary(self):
        summary = self.get_summary_by_channel()
        if not summary:
            return
        
        total = len(self.records)
        uptime = time.time() - self.start_time
        
        print(f"\n{'━'*70}")
        print(f" MÉTRICAS ASCON EN VIVO | {self.device_name} | {total} ops | uptime: {uptime:.0f}s")
        print(f"{'━'*70}")
        print(f" {'Canal':<22} {'#Enc':>5} {'Enc(ms)':>9} {'#Dec':>5} {'Dec(ms)':>9} {'Overhead':>9}")
        print(f"{'─'*70}")
        
        total_enc_ms = 0
        total_dec_ms = 0
        
        for ch_name, ch_data in summary.items():
            n_enc = len(ch_data["enc_times"])
            n_dec = len(ch_data["dec_times"])
            avg_enc = sum(ch_data["enc_times"]) / n_enc if n_enc else 0
            avg_dec = sum(ch_data["dec_times"]) / n_dec if n_dec else 0
            avg_overhead = sum(ch_data["overheads"]) / len(ch_data["overheads"]) if ch_data["overheads"] else 0
            
            total_enc_ms += sum(ch_data["enc_times"])
            total_dec_ms += sum(ch_data["dec_times"])
            
            print(f" {ch_name:<22} {n_enc:>5} {avg_enc:>8.3f}  {n_dec:>5} {avg_dec:>8.3f}  +{avg_overhead:>7.0f}B")
        
        print(f"{'─'*70}")
        print(f" Tiempo total ASCON: {total_enc_ms + total_dec_ms:.1f}ms (enc={total_enc_ms:.1f}ms + dec={total_dec_ms:.1f}ms)")
        print(f"{'━'*70}")
    
    def export_summary(self):
        if not self.records:
            return
        
        self.print_live_summary()
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.records[0].keys())
            writer.writeheader()
            writer.writerows(self.records)
        
        print(f"\n[METRICS] {len(self.records)} registros exportados a {self.csv_path}")
