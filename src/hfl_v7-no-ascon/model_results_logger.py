"""
=============================================================================
 model_results_logger.py — Persistencia CSV para métricas de entrenamiento
=============================================================================
 Guarda accuracy/loss por gateway en una carpeta separada:

   results_model/

 Patrón de salida:
   model_metrics_<device>_<suffix>_<n>.csv
=============================================================================
"""

from __future__ import annotations

import atexit
import csv
import os
import re
import socket
from datetime import datetime
from pathlib import Path


RESULTS_MODEL_DIR = Path(__file__).resolve().parent / "results_model"


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "unknown"


def _next_results_csv_path(device_name: str, suffix: str) -> Path:
    RESULTS_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"model_metrics_{_slug(device_name)}_{_slug(suffix)}_"
    indices = []

    for path in RESULTS_MODEL_DIR.glob(f"{prefix}*.csv"):
        tail = path.stem.replace(prefix, "")
        if tail.isdigit():
            indices.append(int(tail))

    next_index = (max(indices) + 1) if indices else 1
    return RESULTS_MODEL_DIR / f"{prefix}{next_index}.csv"


class ModelResultsLogger:
    def __init__(self, device_name: str = "gateway", suffix: str | None = None):
        hostname = _slug(socket.gethostname())
        env_suffix = os.environ.get("GATEWAY_ID") or os.environ.get("HOSTNAME")

        self.device_name = _slug(device_name)
        self.device_suffix = _slug(suffix or env_suffix or hostname)
        self.hostname = hostname
        self.rows: list[dict] = []
        self.csv_path = _next_results_csv_path(self.device_name, self.device_suffix)
        self.run_id = self.csv_path.stem.replace(f"model_metrics_{self.device_name}_{self.device_suffix}_", "")
        atexit.register(self.flush)

    def record(
        self,
        *,
        stage: str,
        fl_round: int,
        num_samples: int,
        accuracy: float,
        loss: float,
        **extra_fields,
    ) -> None:
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "device_name": self.device_name,
            "device_suffix": self.device_suffix,
            "hostname": self.hostname,
            "run_id": self.run_id,
            "stage": stage,
            "fl_round": fl_round,
            "num_samples": int(num_samples),
            "accuracy": float(accuracy),
            "loss": float(loss),
        }
        row.update(extra_fields)
        self.rows.append(row)
        self.flush()

    def flush(self) -> None:
        if not self.rows:
            return

        fieldnames: list[str] = []
        for row in self.rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
