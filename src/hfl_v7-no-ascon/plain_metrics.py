"""
=============================================================================
 plain_metrics.py — Recolector de métricas sin ASCON
=============================================================================
 Registra tiempos de serialización / parseo de JSON plano para usar como
 baseline frente a las métricas criptográficas de la versión con ASCON.
=============================================================================
"""

from __future__ import annotations

import atexit
import csv
import os
import re
import socket
import time
from datetime import datetime
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parent / "Results"


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "unknown"


def _next_results_csv_path(device_name: str, suffix: str | None = None) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"plain_metrics_{_slug(device_name)}_"
    if suffix:
        prefix += f"{_slug(suffix)}_"

    indices = []
    for path in RESULTS_DIR.glob(f"{prefix}*.csv"):
        tail = path.stem.replace(prefix, "")
        if tail.isdigit():
            indices.append(int(tail))

    next_index = (max(indices) + 1) if indices else 1
    return RESULTS_DIR / f"{prefix}{next_index}.csv"


class PlainMetrics:
    def __init__(self, device_name="unknown", suffix: str | None = None):
        hostname = _slug(socket.gethostname())
        env_suffix = os.environ.get("GATEWAY_ID") or os.environ.get("HOSTNAME")
        self.device_name = _slug(device_name)
        self.device_suffix = _slug(suffix or env_suffix or hostname)
        self.hostname = hostname
        self.records = []
        self.start_time = time.time()
        self.csv_path = _next_results_csv_path(self.device_name, self.device_suffix)
        self.run_id = self.csv_path.stem.replace(f"plain_metrics_{self.device_name}_{self.device_suffix}_", "")
        atexit.register(self.export_summary)

    def record(self, channel, operation, payload_size, elapsed_ms, fl_round, **extra_fields):
        entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "uptime_s": round(time.time() - self.start_time, 1),
            "device_name": self.device_name,
            "device_suffix": self.device_suffix,
            "hostname": self.hostname,
            "run_id": self.run_id,
            "channel": channel,
            "operation": operation,
            "payload_bytes": payload_size,
            "elapsed_ms": round(elapsed_ms, 3),
            "fl_round": fl_round,
        }
        entry.update(extra_fields)
        self.records.append(entry)

        print(
            f"  [PLAIN {operation.upper()}] {channel}: "
            f"{elapsed_ms:.3f}ms | payload={payload_size}B"
        )

    def export_summary(self):
        if not self.records:
            return

        fieldnames = []
        for record in self.records:
            for key in record.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)

        print(f"\n[PLAIN METRICS] {len(self.records)} registros exportados a {self.csv_path}")
