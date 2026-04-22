"""
Herramientas de analisis SRE para hfl_v7-no-ascon usando los CSV historicos ya
guardados en Results/#N. No modifica los archivos fuente del experimento.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd


ATTEMPT_RE = re.compile(r"#(\d+)$")

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-darkgrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 200)
warnings.filterwarnings(
    "ignore",
    message="DataFrameGroupBy.apply operated on the grouping columns",
    category=DeprecationWarning,
)


@dataclass(frozen=True)
class AnalysisConfig:
    results_root: Path
    output_root: Path
    attempt_start: int = 16
    attempt_end: int = 29
    expected_rounds_per_attempt: int = 30
    experiment_name: str = "hfl_v7-no-ascon"


def _safe_pct(series: pd.Series, q: float) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(np.percentile(clean.to_numpy(dtype=float), q))


def _safe_mean(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(clean.mean())


def _safe_std(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(clean.std(ddof=0))


def _discover_attempt_dirs(config: AnalysisConfig) -> list[tuple[int, Path]]:
    if not config.results_root.exists():
        raise FileNotFoundError(f"No existe Results root: {config.results_root}")

    discovered: list[tuple[int, Path]] = []
    for path in config.results_root.iterdir():
        if not path.is_dir():
            continue
        match = ATTEMPT_RE.match(path.name)
        if not match:
            continue
        attempt_id = int(match.group(1))
        if config.attempt_start <= attempt_id <= config.attempt_end:
            discovered.append((attempt_id, path))

    if not discovered:
        raise FileNotFoundError(
            f"No se encontraron intentos entre #{config.attempt_start} y #{config.attempt_end} en {config.results_root}"
        )

    return sorted(discovered, key=lambda item: item[0])


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def _first_existing(attempt_dir: Path, pattern: str) -> Path | None:
    matches = sorted(attempt_dir.glob(pattern))
    return matches[0] if matches else None


def _parse_time_only(series: pd.Series, anchor_date: pd.Timestamp) -> pd.Series:
    anchor_str = anchor_date.strftime("%Y-%m-%d")
    return pd.to_datetime(anchor_str + " " + series.astype(str), errors="coerce")


def _load_attempt_frames(attempt_id: int, attempt_dir: Path) -> dict[str, pd.DataFrame]:
    gateway_transport_paths = sorted(attempt_dir.glob("plain_metrics_gateway_*.csv"))
    server_transport_path = _first_existing(attempt_dir, "plain_metrics_server_*.csv")
    model_metric_paths = sorted(attempt_dir.glob("model_metrics_gateway_*.csv"))
    global_history_path = _first_existing(attempt_dir, "global_weights_history_*.csv")

    gateway_transport = []
    for path in gateway_transport_paths:
        df = _read_csv(path)
        if df.empty:
            continue
        df["attempt_id"] = attempt_id
        df["attempt_dir"] = str(attempt_dir)
        df["source_file"] = path.name
        df["source_scope"] = "gateway_transport"
        gateway_transport.append(df)

    server_transport = _read_csv(server_transport_path) if server_transport_path else pd.DataFrame()
    if not server_transport.empty:
        server_transport["attempt_id"] = attempt_id
        server_transport["attempt_dir"] = str(attempt_dir)
        server_transport["source_file"] = server_transport_path.name
        server_transport["source_scope"] = "server_transport"

    model_metrics = []
    for path in model_metric_paths:
        df = _read_csv(path)
        if df.empty:
            continue
        df["attempt_id"] = attempt_id
        df["attempt_dir"] = str(attempt_dir)
        df["source_file"] = path.name
        df["source_scope"] = "local_training"
        model_metrics.append(df)

    global_history = _read_csv(global_history_path) if global_history_path else pd.DataFrame()
    if not global_history.empty:
        global_history["attempt_id"] = attempt_id
        global_history["attempt_dir"] = str(attempt_dir)
        global_history["source_file"] = global_history_path.name
        global_history["source_scope"] = "global_round"

    gateway_transport_df = pd.concat(gateway_transport, ignore_index=True) if gateway_transport else pd.DataFrame()
    model_metrics_df = pd.concat(model_metrics, ignore_index=True) if model_metrics else pd.DataFrame()

    anchor_date = pd.Timestamp("2026-01-01")
    if not model_metrics_df.empty:
        parsed_local_dt = pd.to_datetime(model_metrics_df["timestamp"], errors="coerce")
        if parsed_local_dt.notna().any():
            anchor_date = parsed_local_dt.dropna().min().normalize()
            model_metrics_df["timestamp_dt"] = parsed_local_dt
    if "timestamp_dt" not in model_metrics_df.columns:
        model_metrics_df["timestamp_dt"] = pd.to_datetime(model_metrics_df.get("timestamp"), errors="coerce")

    if not gateway_transport_df.empty:
        gateway_transport_df["timestamp_dt"] = _parse_time_only(gateway_transport_df["timestamp"], anchor_date)
        gateway_transport_df["event_type"] = "transport.plain"
        gateway_transport_df["round_ref"] = pd.to_numeric(gateway_transport_df["fl_round"], errors="coerce") + 1
        gateway_transport_df["gateway_id"] = gateway_transport_df["device_suffix"].astype(str)
        gateway_transport_df["node_id"] = gateway_transport_df["device_suffix"].astype(str)
    else:
        gateway_transport_df["timestamp_dt"] = pd.Series(dtype="datetime64[ns]")

    if not server_transport.empty:
        server_transport["timestamp_dt"] = _parse_time_only(server_transport["timestamp"], anchor_date)
        server_transport["event_type"] = "transport.plain"
        server_transport["round_ref"] = np.where(
            (server_transport["channel"] == "PC->RPi") & (server_transport["operation"] == "serialize"),
            pd.to_numeric(server_transport["fl_round"], errors="coerce"),
            pd.to_numeric(server_transport["fl_round"], errors="coerce") + 1,
        )
        server_transport["gateway_id"] = "aggregate"
        server_transport["node_id"] = server_transport["device_suffix"].astype(str)
    else:
        server_transport["timestamp_dt"] = pd.Series(dtype="datetime64[ns]")

    if not model_metrics_df.empty:
        model_metrics_df["event_type"] = "model.local_train"
        model_metrics_df["round_ref"] = pd.to_numeric(model_metrics_df["fl_round"], errors="coerce") + 1
        model_metrics_df["gateway_id"] = model_metrics_df["device_suffix"].astype(str)
        model_metrics_df["node_id"] = model_metrics_df["device_suffix"].astype(str)

    if not global_history.empty:
        global_history["timestamp_dt"] = _parse_time_only(global_history["time"], anchor_date)
        global_history["event_type"] = "model.global_round"
        global_history["round_ref"] = pd.to_numeric(global_history["round"], errors="coerce")
        global_history["gateway_id"] = "aggregate"
        global_history["node_id"] = "server"

    return {
        "gateway_transport": gateway_transport_df,
        "server_transport": server_transport,
        "model_metrics": model_metrics_df,
        "global_history": global_history,
    }


def load_all_data(config: AnalysisConfig) -> dict[str, pd.DataFrame]:
    gateway_transport_parts: list[pd.DataFrame] = []
    server_transport_parts: list[pd.DataFrame] = []
    model_metric_parts: list[pd.DataFrame] = []
    global_history_parts: list[pd.DataFrame] = []
    inventory_rows: list[dict[str, Any]] = []

    for attempt_id, attempt_dir in _discover_attempt_dirs(config):
        frames = _load_attempt_frames(attempt_id, attempt_dir)
        gateway_transport_parts.append(frames["gateway_transport"])
        server_transport_parts.append(frames["server_transport"])
        model_metric_parts.append(frames["model_metrics"])
        global_history_parts.append(frames["global_history"])

        inventory_rows.append(
            {
                "attempt_id": attempt_id,
                "attempt_dir": str(attempt_dir),
                "gateway_transport_files": int(len(list(attempt_dir.glob("plain_metrics_gateway_*.csv")))),
                "server_transport_files": int(len(list(attempt_dir.glob("plain_metrics_server_*.csv")))),
                "model_metric_files": int(len(list(attempt_dir.glob("model_metrics_gateway_*.csv")))),
                "global_history_files": int(len(list(attempt_dir.glob("global_weights_history_*.csv")))),
                "pdf_reports": int(len(list(attempt_dir.glob("*.pdf")))),
            }
        )

    gateway_transport = pd.concat(gateway_transport_parts, ignore_index=True) if gateway_transport_parts else pd.DataFrame()
    server_transport = pd.concat(server_transport_parts, ignore_index=True) if server_transport_parts else pd.DataFrame()
    model_metrics = pd.concat(model_metric_parts, ignore_index=True) if model_metric_parts else pd.DataFrame()
    global_history = pd.concat(global_history_parts, ignore_index=True) if global_history_parts else pd.DataFrame()
    attempt_inventory = pd.DataFrame(inventory_rows).sort_values("attempt_id").reset_index(drop=True)

    return {
        "attempt_inventory": attempt_inventory,
        "gateway_transport": gateway_transport,
        "server_transport": server_transport,
        "model_metrics": model_metrics,
        "global_history": global_history,
    }


def build_canonical_log_events(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    events: list[pd.DataFrame] = []

    gateway_transport = data["gateway_transport"].copy()
    if not gateway_transport.empty:
        gateway_transport["event_name"] = "transport.plain.gateway"
        gateway_transport["status"] = "reconstructed_from_csv"
        gateway_transport["trace_id"] = (
            "attempt:"
            + gateway_transport["attempt_id"].astype(str)
            + ":round:"
            + gateway_transport["round_ref"].astype("Int64").astype(str)
        )
        events.append(gateway_transport)

    server_transport = data["server_transport"].copy()
    if not server_transport.empty:
        server_transport["event_name"] = "transport.plain.server"
        server_transport["status"] = "reconstructed_from_csv"
        server_transport["trace_id"] = (
            "attempt:"
            + server_transport["attempt_id"].astype(str)
            + ":round:"
            + server_transport["round_ref"].astype("Int64").astype(str)
        )
        events.append(server_transport)

    model_metrics = data["model_metrics"].copy()
    if not model_metrics.empty:
        model_metrics["event_name"] = "model.local_train"
        model_metrics["status"] = "reconstructed_from_csv"
        model_metrics["trace_id"] = (
            "attempt:"
            + model_metrics["attempt_id"].astype(str)
            + ":round:"
            + model_metrics["round_ref"].astype("Int64").astype(str)
        )
        events.append(model_metrics)

    global_history = data["global_history"].copy()
    if not global_history.empty:
        global_history["event_name"] = "model.global_round"
        global_history["status"] = "reconstructed_from_csv"
        global_history["trace_id"] = (
            "attempt:"
            + global_history["attempt_id"].astype(str)
            + ":round:"
            + global_history["round_ref"].astype("Int64").astype(str)
        )
        events.append(global_history)

    canonical = pd.concat(events, ignore_index=True, sort=False) if events else pd.DataFrame()
    if not canonical.empty:
        canonical = canonical.sort_values(["attempt_id", "timestamp_dt", "event_name"]).reset_index(drop=True)
    return canonical


def build_metric_catalog() -> pd.DataFrame:
    rows = [
        {
            "metric_name": "transport.edge_rpi.deserialize.p95_ms",
            "exactness": "exact_from_csv",
            "source": "plain_metrics_gateway_*",
            "aggregation": "p95(elapsed_ms) WHERE channel='ESP32->RPi' AND operation='deserialize'",
            "purpose": "SLI de latencia de ingreso al gateway por muestra.",
        },
        {
            "metric_name": "transport.rpi_pc.deserialize.p95_ms",
            "exactness": "exact_from_csv",
            "source": "plain_metrics_server_*",
            "aggregation": "p95(elapsed_ms) WHERE channel='RPi->PC' AND operation='deserialize'",
            "purpose": "SLI de latencia de deserializacion en servidor para actualizaciones de gateways.",
        },
        {
            "metric_name": "transport.pc_rpi.serialize.p95_ms",
            "exactness": "exact_from_csv",
            "source": "plain_metrics_server_*",
            "aggregation": "p95(elapsed_ms) WHERE channel='PC->RPi' AND operation='serialize'",
            "purpose": "SLI de despliegue del modelo global desde el servidor.",
        },
        {
            "metric_name": "transport.payload.avg_bytes",
            "exactness": "exact_from_csv",
            "source": "plain_metrics_gateway_*, plain_metrics_server_*",
            "aggregation": "avg(payload_bytes)",
            "purpose": "Tamano promedio del payload transmitido sin cifrado ASCON.",
        },
        {
            "metric_name": "training.local.accuracy.avg",
            "exactness": "exact_from_csv",
            "source": "model_metrics_gateway_*",
            "aggregation": "avg(accuracy) WHERE stage='local_train'",
            "purpose": "Calidad media del entrenamiento local por gateway.",
        },
        {
            "metric_name": "training.local.loss.avg",
            "exactness": "exact_from_csv",
            "source": "model_metrics_gateway_*",
            "aggregation": "avg(loss) WHERE stage='local_train'",
            "purpose": "Estabilidad promedio del entrenamiento local.",
        },
        {
            "metric_name": "training.gateway.accuracy.skew.avg",
            "exactness": "exact_from_csv",
            "source": "model_metrics_gateway_A/B",
            "aggregation": "avg(abs(acc_gateway_A - acc_gateway_B)) por round",
            "purpose": "Desalineacion entre gateways por ronda.",
        },
        {
            "metric_name": "training.gateway.loss.skew.avg",
            "exactness": "exact_from_csv",
            "source": "model_metrics_gateway_A/B",
            "aggregation": "avg(abs(loss_gateway_A - loss_gateway_B)) por round",
            "purpose": "Desalineacion de optimizacion entre gateways.",
        },
        {
            "metric_name": "round.global.accuracy.last",
            "exactness": "exact_from_csv",
            "source": "global_weights_history_*",
            "aggregation": "last(accuracy) por intento",
            "purpose": "Resultado final del modelo global por corrida.",
        },
        {
            "metric_name": "round.global.loss.last",
            "exactness": "exact_from_csv",
            "source": "global_weights_history_*",
            "aggregation": "last(loss) por intento",
            "purpose": "Punto final de convergencia del modelo global.",
        },
        {
            "metric_name": "round.duration.avg_sec",
            "exactness": "exact_from_csv",
            "source": "global_weights_history_*",
            "aggregation": "avg(diff(time)) por intento",
            "purpose": "Duracion promedio de ronda federada.",
        },
        {
            "metric_name": "round.weight_drift.avg",
            "exactness": "exact_from_csv",
            "source": "global_weights_history_*",
            "aggregation": "avg(abs(delta(w3_mag))+abs(delta(w4_*))) por round",
            "purpose": "Movimiento medio de los pesos globales por ronda.",
        },
        {
            "metric_name": "reliability.round_completion.rate",
            "exactness": "exact_from_csv_with_assumption",
            "source": "global_weights_history_*",
            "aggregation": "observed_rounds / expected_rounds_per_attempt",
            "purpose": "Disponibilidad experimental por corrida.",
        },
        {
            "metric_name": "reliability.gateway_participation.rate",
            "exactness": "exact_from_csv",
            "source": "model_metrics_gateway_*",
            "aggregation": "local_train_rounds / global_rounds_observed",
            "purpose": "Tasa de participacion efectiva por gateway.",
        },
    ]
    return pd.DataFrame(rows)


def build_log_catalog() -> pd.DataFrame:
    rows = [
        {
            "log_event": "transport.plain.gateway",
            "status": "reconstructed_from_csv",
            "source": "plain_metrics_gateway_*",
            "fields": "timestamp_dt, attempt_id, gateway_id, channel, operation, elapsed_ms, payload_bytes, client_id, sample_label_name",
            "notes": "Log normalizado derivado sin tocar el CSV original.",
        },
        {
            "log_event": "transport.plain.server",
            "status": "reconstructed_from_csv",
            "source": "plain_metrics_server_*",
            "fields": "timestamp_dt, attempt_id, channel, operation, elapsed_ms, payload_bytes, round_ref",
            "notes": "No contiene gateway_id explicito en los datos actuales.",
        },
        {
            "log_event": "model.local_train",
            "status": "reconstructed_from_csv",
            "source": "model_metrics_gateway_*",
            "fields": "timestamp_dt, attempt_id, gateway_id, fl_round, round_ref, num_samples, accuracy, loss, buffer_target",
            "notes": "Representa el fin del entrenamiento local por gateway.",
        },
        {
            "log_event": "model.global_round",
            "status": "reconstructed_from_csv",
            "source": "global_weights_history_*",
            "fields": "timestamp_dt, attempt_id, round_ref, accuracy, loss, w3_mag, w4_normal, w4_brute, w4_scan",
            "notes": "Evento canonico de cierre de ronda global.",
        },
        {
            "log_event": "fedavg.compute",
            "status": "live_stdout_only",
            "source": "server_hfl.py",
            "fields": "trace_id, ts_start, ts_end, round_ref, gateways_received, fedavg_ms",
            "notes": "No reconstruible con CSV actuales; recomendado solo a consola JSON.",
        },
        {
            "log_event": "model.deploy.gateway",
            "status": "live_stdout_only",
            "source": "gateway_hfl.py",
            "fields": "trace_id, ts, gateway_id, round_ref, payload_bytes, apply_ms, status",
            "notes": "Hoy solo puede estimarse desde transport serialize del servidor.",
        },
    ]
    return pd.DataFrame(rows)


def build_trace_catalog() -> pd.DataFrame:
    rows = [
        {
            "trace_name": "round_trace",
            "status": "reconstructed_from_csv",
            "trace_id_pattern": "attempt:{attempt_id}:round:{round_ref}",
            "spans": "gateway_A.local_train, gateway_B.local_train, server.deserialize_batch, server.serialize_global, global_round_commit",
            "notes": "Traza exacta a nivel de ronda basada en CSV ya capturados.",
        },
        {
            "trace_name": "gateway_round_trace",
            "status": "reconstructed_from_csv",
            "trace_id_pattern": "attempt:{attempt_id}:round:{round_ref}:gateway:{gateway_id}",
            "spans": "esp32_to_rpi_deserialize_batch, local_train_done",
            "notes": "Traza util para diagnosticar carga y skew por gateway.",
        },
        {
            "trace_name": "sample_trace",
            "status": "live_stdout_only",
            "trace_id_pattern": "attempt:{attempt_id}:round:{round_ref}:sample:{client_id}:{sample_seq}",
            "spans": "esp32.publish, gateway.deserialize, gateway.buffer, gateway.train_enqueue",
            "notes": "No reconstruible con precision usando solo CSV actuales.",
        },
        {
            "trace_name": "fedavg_trace",
            "status": "live_stdout_only",
            "trace_id_pattern": "attempt:{attempt_id}:round:{round_ref}:fedavg",
            "spans": "server.wait_updates, server.fedavg_compute, server.serialize_global, server.deploy",
            "notes": "Requiere logging efimero en tiempo real si se quiere exactitud completa.",
        },
    ]
    return pd.DataFrame(rows)


def build_dashboard_catalog() -> pd.DataFrame:
    rows = [
        {
            "panel_group": "System Health",
            "panel_name": "Round completion rate",
            "chart_type": "gauge/bar",
            "source": "global_weights_history_*",
            "metric": "reliability.round_completion.rate",
        },
        {
            "panel_group": "System Health",
            "panel_name": "Gateway participation",
            "chart_type": "bar",
            "source": "model_metrics_gateway_*",
            "metric": "reliability.gateway_participation.rate",
        },
        {
            "panel_group": "Transport",
            "panel_name": "Latency p95 by channel",
            "chart_type": "bar",
            "source": "plain_metrics_*",
            "metric": "transport.*.p95_ms",
        },
        {
            "panel_group": "Transport",
            "panel_name": "Payload bytes by channel",
            "chart_type": "bar",
            "source": "plain_metrics_*",
            "metric": "transport.payload.avg_bytes",
        },
        {
            "panel_group": "Local Training",
            "panel_name": "Local accuracy by gateway",
            "chart_type": "line",
            "source": "model_metrics_gateway_*",
            "metric": "training.local.accuracy.avg",
        },
        {
            "panel_group": "Local Training",
            "panel_name": "Gateway skew",
            "chart_type": "line",
            "source": "model_metrics_gateway_*",
            "metric": "training.gateway.*.skew.avg",
        },
        {
            "panel_group": "Global Convergence",
            "panel_name": "Global accuracy and loss",
            "chart_type": "line",
            "source": "global_weights_history_*",
            "metric": "round.global.*",
        },
        {
            "panel_group": "Global Convergence",
            "panel_name": "Weight drift",
            "chart_type": "line",
            "source": "global_weights_history_*",
            "metric": "round.weight_drift.avg",
        },
        {
            "panel_group": "Data Quality",
            "panel_name": "Class mix by gateway",
            "chart_type": "stacked_bar",
            "source": "plain_metrics_gateway_*",
            "metric": "data.class_mix.share",
        },
    ]
    return pd.DataFrame(rows)


def compute_transport_summary(transport_events: pd.DataFrame) -> pd.DataFrame:
    if transport_events.empty:
        return pd.DataFrame()

    df = transport_events.copy()

    summary = (
        df.groupby(["attempt_id", "source_scope", "device_suffix", "channel", "operation"], dropna=False)
        .apply(
            lambda group: pd.Series(
                {
                    "events": int(len(group)),
                    "avg_ms": _safe_mean(group["elapsed_ms"]),
                    "median_ms": _safe_pct(group["elapsed_ms"], 50),
                    "p95_ms": _safe_pct(group["elapsed_ms"], 95),
                    "max_ms": pd.to_numeric(group["elapsed_ms"], errors="coerce").max(),
                    "avg_payload_bytes": _safe_mean(group["payload_bytes"]),
                    "min_payload_bytes": pd.to_numeric(group["payload_bytes"], errors="coerce").min(),
                    "max_payload_bytes": pd.to_numeric(group["payload_bytes"], errors="coerce").max(),
                }
            )
        )
        .reset_index()
        .sort_values(["attempt_id", "source_scope", "device_suffix", "channel", "operation"])
        .reset_index(drop=True)
    )
    return summary


def compute_local_training_summary(model_metrics: pd.DataFrame) -> pd.DataFrame:
    if model_metrics.empty:
        return pd.DataFrame()

    df = model_metrics.loc[model_metrics["stage"] == "local_train"].copy()
    summary = (
        df.groupby(["attempt_id", "device_suffix"], dropna=False)
        .apply(
            lambda group: pd.Series(
                {
                    "rounds_seen": int(group["fl_round"].nunique()),
                    "avg_accuracy": _safe_mean(group["accuracy"]),
                    "std_accuracy": _safe_std(group["accuracy"]),
                    "min_accuracy": pd.to_numeric(group["accuracy"], errors="coerce").min(),
                    "max_accuracy": pd.to_numeric(group["accuracy"], errors="coerce").max(),
                    "avg_loss": _safe_mean(group["loss"]),
                    "std_loss": _safe_std(group["loss"]),
                    "min_loss": pd.to_numeric(group["loss"], errors="coerce").min(),
                    "max_loss": pd.to_numeric(group["loss"], errors="coerce").max(),
                    "avg_num_samples": _safe_mean(group["num_samples"]),
                    "avg_buffer_target": _safe_mean(group["buffer_target"]),
                }
            )
        )
        .reset_index()
        .sort_values(["attempt_id", "device_suffix"])
        .reset_index(drop=True)
    )
    return summary


def compute_global_round_summary(global_history: pd.DataFrame, expected_rounds_per_attempt: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if global_history.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = global_history.copy().sort_values(["attempt_id", "round"])
    df["round_duration_sec"] = df.groupby("attempt_id")["timestamp_dt"].diff().dt.total_seconds()
    df["w3_drift_abs"] = df.groupby("attempt_id")["w3_mag"].diff().abs()
    df["w4_normal_drift_abs"] = df.groupby("attempt_id")["w4_normal"].diff().abs()
    df["w4_brute_drift_abs"] = df.groupby("attempt_id")["w4_brute"].diff().abs()
    df["w4_scan_drift_abs"] = df.groupby("attempt_id")["w4_scan"].diff().abs()
    df["weight_drift_total_abs"] = (
        df[["w3_drift_abs", "w4_normal_drift_abs", "w4_brute_drift_abs", "w4_scan_drift_abs"]]
        .fillna(0.0)
        .sum(axis=1)
    )

    summary = (
        df.groupby("attempt_id", dropna=False)
        .apply(
            lambda group: pd.Series(
                {
                    "rounds_observed": int(group["round"].nunique()),
                    "completion_rate": float(group["round"].nunique() / expected_rounds_per_attempt),
                    "last_accuracy": pd.to_numeric(group["accuracy"], errors="coerce").iloc[-1],
                    "last_loss": pd.to_numeric(group["loss"], errors="coerce").iloc[-1],
                    "best_accuracy": pd.to_numeric(group["accuracy"], errors="coerce").max(),
                    "min_loss": pd.to_numeric(group["loss"], errors="coerce").min(),
                    "avg_round_duration_sec": _safe_mean(group["round_duration_sec"]),
                    "p95_round_duration_sec": _safe_pct(group["round_duration_sec"], 95),
                    "avg_weight_drift_total_abs": _safe_mean(group["weight_drift_total_abs"]),
                }
            )
        )
        .reset_index()
        .sort_values("attempt_id")
        .reset_index(drop=True)
    )
    return summary, df


def compute_gateway_alignment(model_metrics: pd.DataFrame, global_history: pd.DataFrame) -> pd.DataFrame:
    if model_metrics.empty:
        return pd.DataFrame()

    local_df = model_metrics.loc[model_metrics["stage"] == "local_train"].copy()
    local_df["global_round"] = pd.to_numeric(local_df["fl_round"], errors="coerce") + 1
    pivot = (
        local_df.pivot_table(
            index=["attempt_id", "global_round"],
            columns="device_suffix",
            values=["accuracy", "loss", "timestamp_dt"],
            aggfunc="first",
        )
        .sort_index()
    )

    gateways = sorted(local_df["device_suffix"].dropna().unique().tolist())
    if len(gateways) < 2:
        return pd.DataFrame()

    first_gateway, second_gateway = gateways[0], gateways[1]
    alignment = pivot.copy()
    alignment.columns = [f"{metric}_{gateway}" for metric, gateway in alignment.columns]
    alignment = alignment.reset_index()
    alignment["accuracy_skew_abs"] = (
        pd.to_numeric(alignment.get(f"accuracy_{first_gateway}"), errors="coerce")
        - pd.to_numeric(alignment.get(f"accuracy_{second_gateway}"), errors="coerce")
    ).abs()
    alignment["loss_skew_abs"] = (
        pd.to_numeric(alignment.get(f"loss_{first_gateway}"), errors="coerce")
        - pd.to_numeric(alignment.get(f"loss_{second_gateway}"), errors="coerce")
    ).abs()
    alignment["gateway_train_skew_sec"] = (
        pd.to_datetime(alignment.get(f"timestamp_dt_{first_gateway}"), errors="coerce")
        - pd.to_datetime(alignment.get(f"timestamp_dt_{second_gateway}"), errors="coerce")
    ).abs().dt.total_seconds()

    if not global_history.empty:
        global_join = global_history[["attempt_id", "round", "accuracy", "loss"]].rename(
            columns={"round": "global_round", "accuracy": "global_accuracy", "loss": "global_loss"}
        )
        alignment = alignment.merge(global_join, on=["attempt_id", "global_round"], how="left")

    return alignment.sort_values(["attempt_id", "global_round"]).reset_index(drop=True)


def compute_class_mix(gateway_transport: pd.DataFrame) -> pd.DataFrame:
    if gateway_transport.empty or "sample_label_name" not in gateway_transport.columns:
        return pd.DataFrame()

    df = gateway_transport.copy()
    counts = (
        df.groupby(["attempt_id", "device_suffix", "sample_label_name"], dropna=False)
        .size()
        .reset_index(name="samples")
    )
    totals = counts.groupby(["attempt_id", "device_suffix"])["samples"].transform("sum")
    counts["share_pct"] = counts["samples"] / totals * 100.0
    return counts.sort_values(["attempt_id", "device_suffix", "sample_label_name"]).reset_index(drop=True)


def build_round_trace_summary(
    gateway_transport: pd.DataFrame,
    server_transport: pd.DataFrame,
    model_metrics: pd.DataFrame,
    global_history: pd.DataFrame,
) -> pd.DataFrame:
    if global_history.empty:
        return pd.DataFrame()

    local_df = model_metrics.loc[model_metrics["stage"] == "local_train"].copy()
    local_df["global_round"] = pd.to_numeric(local_df["fl_round"], errors="coerce") + 1

    gateway_batch = (
        gateway_transport.groupby(["attempt_id", "device_suffix", "round_ref"], dropna=False)
        .apply(
            lambda group: pd.Series(
                {
                    "gateway_events": int(len(group)),
                    "gateway_deserialize_avg_ms": _safe_mean(group["elapsed_ms"]),
                    "gateway_deserialize_p95_ms": _safe_pct(group["elapsed_ms"], 95),
                    "gateway_avg_payload_bytes": _safe_mean(group["payload_bytes"]),
                    "gateway_samples_seen": int(group["client_id"].notna().sum()) if "client_id" in group else int(len(group)),
                }
            )
        )
        .reset_index()
        .rename(columns={"round_ref": "global_round"})
    )

    local_rounds = (
        local_df.groupby(["attempt_id", "device_suffix", "global_round"], dropna=False)
        .agg(
            local_accuracy=("accuracy", "first"),
            local_loss=("loss", "first"),
            local_num_samples=("num_samples", "first"),
            local_timestamp=("timestamp_dt", "first"),
        )
        .reset_index()
    )

    server_rounds = (
        server_transport.groupby(["attempt_id", "round_ref", "channel", "operation"], dropna=False)
        .apply(
            lambda group: pd.Series(
                {
                    "server_events": int(len(group)),
                    "server_avg_ms": _safe_mean(group["elapsed_ms"]),
                    "server_p95_ms": _safe_pct(group["elapsed_ms"], 95),
                    "server_avg_payload_bytes": _safe_mean(group["payload_bytes"]),
                }
            )
        )
        .reset_index()
        .rename(columns={"round_ref": "global_round"})
    )

    deserialize_server = server_rounds.loc[
        (server_rounds["channel"] == "RPi->PC") & (server_rounds["operation"] == "deserialize")
    ].rename(
        columns={
            "server_events": "server_rpi_pc_events",
            "server_avg_ms": "server_rpi_pc_avg_ms",
            "server_p95_ms": "server_rpi_pc_p95_ms",
            "server_avg_payload_bytes": "server_rpi_pc_avg_payload_bytes",
        }
    )
    serialize_server = server_rounds.loc[
        (server_rounds["channel"] == "PC->RPi") & (server_rounds["operation"] == "serialize")
    ].rename(
        columns={
            "server_events": "server_pc_rpi_events",
            "server_avg_ms": "server_pc_rpi_avg_ms",
            "server_p95_ms": "server_pc_rpi_p95_ms",
            "server_avg_payload_bytes": "server_pc_rpi_avg_payload_bytes",
        }
    )

    traces: list[dict[str, Any]] = []
    for _, row in global_history.sort_values(["attempt_id", "round"]).iterrows():
        attempt_id = int(row["attempt_id"])
        global_round = int(row["round"])
        trace_id = f"attempt:{attempt_id}:round:{global_round}"

        base: dict[str, Any] = {
            "trace_id": trace_id,
            "attempt_id": attempt_id,
            "global_round": global_round,
            "global_timestamp": row["timestamp_dt"],
            "global_accuracy": row["accuracy"],
            "global_loss": row["loss"],
            "w3_mag": row["w3_mag"],
            "w4_normal": row["w4_normal"],
            "w4_brute": row["w4_brute"],
            "w4_scan": row["w4_scan"],
        }

        for gateway_id in sorted(local_df["device_suffix"].dropna().unique().tolist()):
            local_row = local_rounds.loc[
                (local_rounds["attempt_id"] == attempt_id)
                & (local_rounds["device_suffix"] == gateway_id)
                & (local_rounds["global_round"] == global_round)
            ]
            gateway_row = gateway_batch.loc[
                (gateway_batch["attempt_id"] == attempt_id)
                & (gateway_batch["device_suffix"] == gateway_id)
                & (gateway_batch["global_round"] == global_round)
            ]
            if not local_row.empty:
                base[f"{gateway_id}_local_accuracy"] = local_row["local_accuracy"].iloc[0]
                base[f"{gateway_id}_local_loss"] = local_row["local_loss"].iloc[0]
                base[f"{gateway_id}_local_num_samples"] = local_row["local_num_samples"].iloc[0]
                base[f"{gateway_id}_local_timestamp"] = local_row["local_timestamp"].iloc[0]
            if not gateway_row.empty:
                base[f"{gateway_id}_gateway_events"] = gateway_row["gateway_events"].iloc[0]
                base[f"{gateway_id}_gateway_deserialize_avg_ms"] = gateway_row["gateway_deserialize_avg_ms"].iloc[0]
                base[f"{gateway_id}_gateway_deserialize_p95_ms"] = gateway_row["gateway_deserialize_p95_ms"].iloc[0]
                base[f"{gateway_id}_gateway_avg_payload_bytes"] = gateway_row["gateway_avg_payload_bytes"].iloc[0]
                base[f"{gateway_id}_gateway_samples_seen"] = gateway_row["gateway_samples_seen"].iloc[0]

        server_dec = deserialize_server.loc[
            (deserialize_server["attempt_id"] == attempt_id) & (deserialize_server["global_round"] == global_round)
        ]
        if not server_dec.empty:
            for col in [
                "server_rpi_pc_events",
                "server_rpi_pc_avg_ms",
                "server_rpi_pc_p95_ms",
                "server_rpi_pc_avg_payload_bytes",
            ]:
                base[col] = server_dec[col].iloc[0]

        server_enc = serialize_server.loc[
            (serialize_server["attempt_id"] == attempt_id) & (serialize_server["global_round"] == global_round)
        ]
        if not server_enc.empty:
            for col in [
                "server_pc_rpi_events",
                "server_pc_rpi_avg_ms",
                "server_pc_rpi_p95_ms",
                "server_pc_rpi_avg_payload_bytes",
            ]:
                base[col] = server_enc[col].iloc[0]

        traces.append(base)

    trace_df = pd.DataFrame(traces).sort_values(["attempt_id", "global_round"]).reset_index(drop=True)
    return trace_df


def compute_executive_summary(
    attempt_inventory: pd.DataFrame,
    transport_summary: pd.DataFrame,
    local_training_summary: pd.DataFrame,
    global_round_summary: pd.DataFrame,
    class_mix: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "attempts_analyzed": int(len(attempt_inventory)),
        "attempt_range": [
            int(attempt_inventory["attempt_id"].min()) if not attempt_inventory.empty else None,
            int(attempt_inventory["attempt_id"].max()) if not attempt_inventory.empty else None,
        ],
        "avg_gateway_transport_p95_ms": _safe_mean(
            transport_summary.loc[
                (transport_summary["source_scope"] == "gateway_transport")
                & (transport_summary["channel"] == "ESP32->RPi")
                & (transport_summary["operation"] == "deserialize"),
                "p95_ms",
            ]
        ),
        "avg_server_deserialize_p95_ms": _safe_mean(
            transport_summary.loc[
                (transport_summary["source_scope"] == "server_transport")
                & (transport_summary["channel"] == "RPi->PC")
                & (transport_summary["operation"] == "deserialize"),
                "p95_ms",
            ]
        ),
        "avg_payload_bytes": _safe_mean(transport_summary["avg_payload_bytes"]),
        "avg_local_accuracy": _safe_mean(local_training_summary["avg_accuracy"]),
        "avg_local_loss": _safe_mean(local_training_summary["avg_loss"]),
        "avg_global_last_accuracy": _safe_mean(global_round_summary["last_accuracy"]),
        "avg_global_last_loss": _safe_mean(global_round_summary["last_loss"]),
        "avg_round_completion_rate": _safe_mean(global_round_summary["completion_rate"]),
        "avg_round_duration_sec": _safe_mean(global_round_summary["avg_round_duration_sec"]),
        "class_labels_observed": sorted(class_mix["sample_label_name"].dropna().astype(str).unique().tolist())
        if not class_mix.empty
        else [],
    }


def _save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def _save_plot(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_Sin datos_"

    columns = [str(col) for col in df.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            value = row[col]
            if pd.isna(value):
                values.append("")
            else:
                values.append(str(value).replace("\n", " "))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def generate_plots(
    output_dir: Path,
    transport_summary: pd.DataFrame,
    alignment: pd.DataFrame,
    global_history_enriched: pd.DataFrame,
    class_mix: pd.DataFrame,
    attempt_start: int,
    attempt_end: int,
) -> dict[str, str]:
    plot_paths: dict[str, str] = {}

    if not transport_summary.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        chart = transport_summary.groupby(["source_scope", "channel", "operation"])["p95_ms"].mean().reset_index()
        chart["label"] = chart["source_scope"] + " | " + chart["channel"] + " | " + chart["operation"]
        ax.bar(chart["label"], chart["p95_ms"], color="#1f77b4")
        ax.set_title(f"Latency p95 promedio por canal y operacion (Intentos #{attempt_start}-#{attempt_end})")
        ax.set_xlabel("Canal")
        ax.set_ylabel("p95 latency (ms)")
        ax.tick_params(axis="x", rotation=45)
        path = output_dir / "transport_latency_p95.png"
        _save_plot(fig, path)
        plot_paths["transport_latency_p95"] = str(path)

        fig, ax = plt.subplots(figsize=(10, 5))
        payload = transport_summary.groupby(["source_scope", "channel"])["avg_payload_bytes"].mean().reset_index()
        payload["label"] = payload["source_scope"] + " | " + payload["channel"]
        ax.bar(payload["label"], payload["avg_payload_bytes"], color="#ff7f0e")
        ax.set_title("Tamano promedio de payload por canal")
        ax.set_xlabel("Canal")
        ax.set_ylabel("Payload promedio (bytes)")
        ax.tick_params(axis="x", rotation=45)
        path = output_dir / "transport_payload_bytes.png"
        _save_plot(fig, path)
        plot_paths["transport_payload_bytes"] = str(path)

    if not global_history_enriched.empty:
        fig, ax1 = plt.subplots(figsize=(11, 5))
        by_round = global_history_enriched.groupby("round")[["accuracy", "loss"]].mean().reset_index()
        ax1.plot(by_round["round"], by_round["accuracy"], marker="o", label="Global Accuracy", color="#2ca02c")
        ax1.set_xlabel("Ronda global")
        ax1.set_ylabel("Accuracy", color="#2ca02c")
        ax1.tick_params(axis="y", labelcolor="#2ca02c")
        ax2 = ax1.twinx()
        ax2.plot(by_round["round"], by_round["loss"], marker="s", label="Global Loss", color="#d62728")
        ax2.set_ylabel("Loss", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        ax1.set_title(f"Convergencia global promedio (#{attempt_start}-#{attempt_end})")
        path = output_dir / "global_accuracy_loss.png"
        _save_plot(fig, path)
        plot_paths["global_accuracy_loss"] = str(path)

        fig, ax = plt.subplots(figsize=(11, 5))
        weights = global_history_enriched.groupby("round")[["w3_mag", "w4_normal", "w4_brute", "w4_scan"]].mean().reset_index()
        for col in ["w3_mag", "w4_normal", "w4_brute", "w4_scan"]:
            ax.plot(weights["round"], weights[col], marker="o", label=col)
        ax.set_title("Magnitud promedio de pesos globales")
        ax.set_xlabel("Ronda global")
        ax.set_ylabel("Magnitud")
        ax.legend()
        path = output_dir / "weight_magnitude_trends.png"
        _save_plot(fig, path)
        plot_paths["weight_magnitude_trends"] = str(path)

        fig, ax = plt.subplots(figsize=(11, 5))
        duration = global_history_enriched.groupby("round")["round_duration_sec"].mean().reset_index()
        duration = duration.dropna()
        ax.plot(duration["round"], duration["round_duration_sec"], marker="o", color="#9467bd")
        ax.set_title("Duracion promedio por ronda")
        ax.set_xlabel("Ronda global")
        ax.set_ylabel("Duracion promedio (s)")
        path = output_dir / "round_duration.png"
        _save_plot(fig, path)
        plot_paths["round_duration"] = str(path)

    if not alignment.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        skew = alignment.groupby("global_round")["accuracy_skew_abs"].mean().reset_index()
        ax.plot(skew["global_round"], skew["accuracy_skew_abs"], marker="o", color="#8c564b")
        ax.set_title("Skew promedio de accuracy entre gateways")
        ax.set_xlabel("Ronda global")
        ax.set_ylabel("|acc_A - acc_B|")
        path = output_dir / "gateway_accuracy_skew.png"
        _save_plot(fig, path)
        plot_paths["gateway_accuracy_skew"] = str(path)

    if not class_mix.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        share = (
            class_mix.groupby(["device_suffix", "sample_label_name"])["samples"]
            .sum()
            .reset_index()
            .pivot(index="device_suffix", columns="sample_label_name", values="samples")
            .fillna(0)
        )
        share.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("Distribucion total de clases observadas por gateway")
        ax.set_xlabel("Gateway")
        ax.set_ylabel("Muestras")
        ax.legend(title="Clase")
        path = output_dir / "class_mix_by_gateway.png"
        _save_plot(fig, path)
        plot_paths["class_mix_by_gateway"] = str(path)

    return plot_paths


def write_spec_markdown(
    output_dir: Path,
    metric_catalog: pd.DataFrame,
    log_catalog: pd.DataFrame,
    trace_catalog: pd.DataFrame,
    dashboard_catalog: pd.DataFrame,
    executive_summary: dict[str, Any],
    experiment_name: str,
) -> Path:
    path = output_dir / "SRE_OBSERVABILITY_SPEC.md"
    lines = [
        f"# SRE Observability Spec - {experiment_name}",
        "",
        f"Generado a partir de `Results/#{executive_summary['attempt_range'][0]}` a `#{executive_summary['attempt_range'][1]}` sin modificar los CSV fuente.",
        "",
        "## Executive Summary",
        "",
        f"- Intentos analizados: `{executive_summary['attempts_analyzed']}`",
        f"- Rango de intentos: `{executive_summary['attempt_range'][0]}` a `{executive_summary['attempt_range'][1]}`",
        f"- p95 promedio ESP32->RPi deserialize: `{executive_summary['avg_gateway_transport_p95_ms']:.3f} ms`",
        f"- p95 promedio RPi->PC deserialize: `{executive_summary['avg_server_deserialize_p95_ms']:.3f} ms`",
        f"- Payload promedio: `{executive_summary['avg_payload_bytes']:.3f} bytes`",
        f"- Accuracy local promedio: `{executive_summary['avg_local_accuracy']:.4f}`",
        f"- Loss local promedio: `{executive_summary['avg_local_loss']:.4f}`",
        f"- Accuracy global final promedio: `{executive_summary['avg_global_last_accuracy']:.4f}`",
        f"- Loss global final promedio: `{executive_summary['avg_global_last_loss']:.4f}`",
        f"- Completion rate promedio: `{executive_summary['avg_round_completion_rate']:.4f}`",
        "",
        "## Metric Catalog",
        "",
        _df_to_markdown_table(metric_catalog),
        "",
        "## Log Catalog",
        "",
        _df_to_markdown_table(log_catalog),
        "",
        "## Trace Catalog",
        "",
        _df_to_markdown_table(trace_catalog),
        "",
        "## Dashboard Panels",
        "",
        _df_to_markdown_table(dashboard_catalog),
        "",
        "## Scope",
        "",
        "- Reconstruible ahora: metricas, logs canonicos y trazas de ronda a partir de CSV historicos.",
        "- Solo live stdout: spans internos de FedAvg, aplicacion del modelo en gateway y trazas por muestra.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_analysis(
    results_root: Path | str,
    output_root: Path | str,
    attempt_start: int = 3,
    attempt_end: int = 11,
    expected_rounds_per_attempt: int = 30,
) -> dict[str, Any]:
    config = AnalysisConfig(
        results_root=Path(results_root),
        output_root=Path(output_root),
        attempt_start=attempt_start,
        attempt_end=attempt_end,
        expected_rounds_per_attempt=expected_rounds_per_attempt,
    )
    output_dir = config.output_root / f"sre_results_{config.attempt_start}_{config.attempt_end}"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_all_data(config)
    transport_events = pd.concat(
        [data["gateway_transport"], data["server_transport"]],
        ignore_index=True,
        sort=False,
    )
    canonical_logs = build_canonical_log_events(data)
    metric_catalog = build_metric_catalog()
    log_catalog = build_log_catalog()
    trace_catalog = build_trace_catalog()
    dashboard_catalog = build_dashboard_catalog()
    transport_summary = compute_transport_summary(transport_events)
    local_training_summary = compute_local_training_summary(data["model_metrics"])
    global_round_summary, global_history_enriched = compute_global_round_summary(
        data["global_history"], config.expected_rounds_per_attempt
    )
    gateway_alignment = compute_gateway_alignment(data["model_metrics"], data["global_history"])
    class_mix = compute_class_mix(data["gateway_transport"])
    round_trace_summary = build_round_trace_summary(
        data["gateway_transport"],
        data["server_transport"],
        data["model_metrics"],
        data["global_history"],
    )
    executive_summary = compute_executive_summary(
        data["attempt_inventory"],
        transport_summary,
        local_training_summary,
        global_round_summary,
        class_mix,
    )
    plot_paths = generate_plots(
        output_dir,
        transport_summary,
        gateway_alignment,
        global_history_enriched,
        class_mix,
        config.attempt_start,
        config.attempt_end,
    )

    tables = {
        "attempt_inventory": data["attempt_inventory"],
        "canonical_log_events": canonical_logs,
        "metric_catalog": metric_catalog,
        "log_catalog": log_catalog,
        "trace_catalog": trace_catalog,
        "dashboard_catalog": dashboard_catalog,
        "transport_sli_summary": transport_summary,
        "local_training_sli_summary": local_training_summary,
        "global_round_sli_summary": global_round_summary,
        "global_round_enriched": global_history_enriched,
        "gateway_alignment_sli_summary": gateway_alignment,
        "class_mix_summary": class_mix,
        "round_trace_summary": round_trace_summary,
    }

    for name, table in tables.items():
        _save_dataframe(table, output_dir / f"{name}.csv")

    (output_dir / "executive_summary.json").write_text(
        json.dumps(executive_summary, indent=2),
        encoding="utf-8",
    )
    spec_path = write_spec_markdown(
        output_dir,
        metric_catalog,
        log_catalog,
        trace_catalog,
        dashboard_catalog,
        executive_summary,
        config.experiment_name,
    )

    return {
        "config": config,
        "output_dir": output_dir,
        "tables": tables,
        "executive_summary": executive_summary,
        "plot_paths": plot_paths,
        "spec_path": spec_path,
    }


if __name__ == "__main__":
    default_results_root = Path(
        r"C:\Users\VivoBook\Downloads\microproyecto2\microproyecto 2 porque wtf\src\hfl_v7-no-ascon\Results"
    )
    default_output_root = Path(
        r"C:\Users\VivoBook\Downloads\microproyecto2\microproyecto 2 porque wtf\src\Analisis de Modelos\no-ascon\analysis_outputs"
    )
    bundle = run_analysis(
        results_root=default_results_root,
        output_root=default_output_root,
        attempt_start=16,
        attempt_end=29,
        expected_rounds_per_attempt=30,
    )
    print(f"Analisis SRE generado en: {bundle['output_dir']}")
