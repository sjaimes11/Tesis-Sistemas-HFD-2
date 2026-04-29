"""
Analisis completo SRE/NIST para experimentos HFL v7.

Lee de forma read-only los CSV historicos guardados en Results/#N y
Results_FOG/#N para RN, CNN, ASCON y no-ASCON. Genera salidas agregadas en:
Analisis de Modelos/Completo/analysis_outputs/complete_hfl_analysis
"""

from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover - el analisis sigue sin scipy.
    stats = None


matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.style.use("seaborn-v0_8-darkgrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)
warnings.filterwarnings(
    "ignore",
    message="DataFrameGroupBy.apply operated on the grouping columns",
    category=DeprecationWarning,
)


ATTEMPT_RE = re.compile(r"#(\d+)$")
EXPECTED_ROUNDS_PER_ATTEMPT = 30
ALPHA = 0.05

# Estos umbrales son criterios operativos del proyecto para reportar evidencia.
# No deben presentarse como umbrales oficiales de aprobacion NIST.
PROJECT_MIN_ASCON_OPS = 1000
PROJECT_LATENCY_P95_LIMIT_MS = 50.0
PROJECT_OVERHEAD_LIMIT_PCT = 50.0
RECOMMENDED_INDEPENDENT_ATTEMPTS = 30

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(__file__).resolve().parent / "analysis_outputs" / "complete_hfl_analysis"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    architecture: str
    security: str
    topology: str
    results_root: Path
    expected_rounds: int = EXPECTED_ROUNDS_PER_ATTEMPT

    @property
    def mode(self) -> str:
        return "ascon" if self.security == "ASCON" else "plain"


EXPERIMENTS = [
    ExperimentConfig("CNN_ASCON", "CNN", "ASCON", "standard", ROOT / "hfl_v7-CNN" / "Results"),
    ExperimentConfig("CNN_FOG_ASCON", "CNN", "ASCON", "fog", ROOT / "hfl_v7-CNN" / "Results_FOG"),
    ExperimentConfig("CNN_NO_ASCON", "CNN", "NO_ASCON", "standard", ROOT / "hfl_v7-no-ascon-CNN" / "Results"),
    ExperimentConfig("CNN_FOG_NO_ASCON", "CNN", "NO_ASCON", "fog", ROOT / "hfl_v7-no-ascon-CNN" / "Results_FOG"),
    ExperimentConfig("RN_ASCON", "RN", "ASCON", "standard", ROOT / "hfl_v7-RN" / "Results"),
    ExperimentConfig("RN_FOG_ASCON", "RN", "ASCON", "fog", ROOT / "hfl_v7-RN" / "Results_FOG"),
    ExperimentConfig("RN_NO_ASCON", "RN", "NO_ASCON", "standard", ROOT / "hfl_v7-no-ascon-RN" / "Results"),
    ExperimentConfig("RN_FOG_NO_ASCON", "RN", "NO_ASCON", "fog", ROOT / "hfl_v7-no-ascon-RN" / "Results_FOG"),
]


def _safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _safe_mean(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.mean()) if not clean.empty else float("nan")


def _safe_std(series: pd.Series, ddof: int = 0) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.std(ddof=ddof)) if len(clean) > ddof else float("nan")


def _safe_pct(series: pd.Series, q: float) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(np.percentile(clean.to_numpy(dtype=float), q))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        print(f"[WARN] No se pudo leer {path}: {exc}")
        return pd.DataFrame()


def _discover_attempt_dirs(config: ExperimentConfig) -> list[tuple[int, Path]]:
    if not config.results_root.exists():
        return []

    attempts: list[tuple[int, Path]] = []
    for path in config.results_root.iterdir():
        if not path.is_dir():
            continue
        match = ATTEMPT_RE.match(path.name)
        if not match:
            continue
        attempts.append((int(match.group(1)), path))
    return sorted(attempts, key=lambda item: item[0])


def _normalize_channel(value: Any) -> str:
    text = str(value) if not pd.isna(value) else ""
    return (
        text.replace("RPi_leader->PC", "RPi->PC")
        .replace("PC->RPi_leader", "PC->RPi")
        .replace("FogLeader->PC", "RPi->PC")
        .replace("PC->FogLeader", "PC->RPi")
    )


def _series_for(frame: pd.DataFrame, column: str, default: Any = pd.NA) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series([default] * len(frame), index=frame.index)


def _numeric_series_for(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(_series_for(frame, column, np.nan), errors="coerce")


def _parse_time_only(series: pd.Series, anchor_date: pd.Timestamp) -> pd.Series:
    anchor = anchor_date.strftime("%Y-%m-%d")
    return pd.to_datetime(anchor + " " + series.astype(str), errors="coerce")


def _infer_anchor_date(*frames: pd.DataFrame) -> pd.Timestamp:
    for frame in frames:
        if frame.empty or "timestamp" not in frame.columns:
            continue
        parsed = pd.to_datetime(frame["timestamp"], errors="coerce")
        if parsed.notna().any():
            return parsed.dropna().min().normalize()
    return pd.Timestamp("2026-01-01")


def _attach_common_metadata(df: pd.DataFrame, config: ExperimentConfig, attempt_id: int, attempt_dir: Path, path: Path) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["experiment"] = config.name
    out["architecture"] = config.architecture
    out["security"] = config.security
    out["topology"] = config.topology
    out["mode"] = config.mode
    out["attempt_id"] = int(attempt_id)
    out["attempt_key"] = f"{config.name}#{attempt_id}"
    out["attempt_dir"] = str(attempt_dir)
    out["results_root"] = str(config.results_root)
    out["source_file"] = path.name
    return out


def _load_attempt(config: ExperimentConfig, attempt_id: int, attempt_dir: Path) -> dict[str, pd.DataFrame]:
    csv_paths = sorted(attempt_dir.glob("*.csv"))
    transport_paths = [
        path
        for path in csv_paths
        if ("ascon_metrics" in path.name.lower() or "plain_metrics" in path.name.lower())
    ]
    model_paths = [path for path in csv_paths if "model_metrics" in path.name.lower()]
    global_paths = [path for path in csv_paths if "global_weights_history" in path.name.lower()]

    raw_models = []
    for path in model_paths:
        frame = _attach_common_metadata(_read_csv(path), config, attempt_id, attempt_dir, path)
        if not frame.empty:
            raw_models.append(frame)
    model_metrics = pd.concat(raw_models, ignore_index=True, sort=False) if raw_models else pd.DataFrame()

    raw_globals = []
    for path in global_paths:
        frame = _attach_common_metadata(_read_csv(path), config, attempt_id, attempt_dir, path)
        if not frame.empty:
            raw_globals.append(frame)
    global_history = pd.concat(raw_globals, ignore_index=True, sort=False) if raw_globals else pd.DataFrame()

    anchor_date = _infer_anchor_date(model_metrics)

    transport_frames = []
    for path in transport_paths:
        frame = _attach_common_metadata(_read_csv(path), config, attempt_id, attempt_dir, path)
        if frame.empty:
            continue
        lower_name = path.name.lower()
        frame["source_scope"] = "server_transport" if "server" in lower_name else "gateway_transport"
        frame["transport_kind"] = "ascon" if "ascon_metrics" in lower_name else "plain"
        frame["channel"] = _series_for(frame, "channel", "").map(_normalize_channel)
        frame["operation"] = _series_for(frame, "operation", "").astype(str).str.lower()
        frame["timestamp_dt"] = _parse_time_only(_series_for(frame, "timestamp", ""), anchor_date)
        frame["event_name"] = np.where(
            frame["source_scope"].eq("server_transport"),
            np.where(frame["transport_kind"].eq("ascon"), "transport.crypto.server", "transport.plain.server"),
            np.where(frame["transport_kind"].eq("ascon"), "transport.crypto.gateway", "transport.plain.gateway"),
        )

        fl_round = _numeric_series_for(frame, "fl_round")
        is_egress = frame["channel"].eq("PC->RPi") & frame["operation"].isin(["encrypt", "serialize"])
        frame["round_ref"] = np.where(is_egress, fl_round, fl_round + 1)
        frame["node_id"] = _series_for(frame, "device_suffix", "unknown").astype(str)
        frame["gateway_id"] = np.where(frame["source_scope"].eq("server_transport"), "aggregate", frame["node_id"])

        payload_bytes = _numeric_series_for(frame, "payload_bytes")
        pt_bytes = _numeric_series_for(frame, "pt_bytes")
        enc_bytes = _numeric_series_for(frame, "enc_bytes")
        overhead = _numeric_series_for(frame, "overhead_bytes")

        frame["payload_plain_bytes"] = pt_bytes.fillna(payload_bytes)
        frame["payload_wire_bytes"] = enc_bytes.fillna(payload_bytes)
        computed_overhead = frame["payload_wire_bytes"] - frame["payload_plain_bytes"]
        frame["overhead_bytes"] = overhead.fillna(computed_overhead).fillna(0.0)
        frame.loc[frame["transport_kind"].eq("plain"), "overhead_bytes"] = 0.0
        frame["elapsed_ms"] = pd.to_numeric(frame.get("elapsed_ms"), errors="coerce")
        frame["overhead_pct"] = frame["overhead_bytes"] / frame["payload_plain_bytes"] * 100.0
        frame["expansion_ratio"] = frame["payload_wire_bytes"] / frame["payload_plain_bytes"]
        transport_frames.append(frame)

    transport = pd.concat(transport_frames, ignore_index=True, sort=False) if transport_frames else pd.DataFrame()

    if not model_metrics.empty:
        model_metrics["timestamp_dt"] = pd.to_datetime(_series_for(model_metrics, "timestamp", ""), errors="coerce")
        model_metrics["stage"] = _series_for(model_metrics, "stage", "").astype(str)
        model_metrics["event_name"] = np.select(
            [
                model_metrics["stage"].eq("fog_fedavg"),
                model_metrics["stage"].eq("global_eval"),
            ],
            [
                "model.fog_fedavg",
                "model.global_eval",
            ],
            default="model.local_train",
        )
        model_metrics["source_scope"] = "model_metrics"
        model_metrics["round_ref"] = _numeric_series_for(model_metrics, "fl_round") + 1
        model_metrics["node_id"] = _series_for(model_metrics, "device_suffix", "unknown").astype(str)
        model_metrics["gateway_id"] = model_metrics["node_id"]

    if not global_history.empty:
        global_history["timestamp_dt"] = _parse_time_only(_series_for(global_history, "time", ""), anchor_date)
        global_history["event_name"] = "model.global_round"
        global_history["source_scope"] = "global_history"
        global_history["round_ref"] = _numeric_series_for(global_history, "round")
        global_history["node_id"] = "server"
        global_history["gateway_id"] = "aggregate"
        for col in ["accuracy", "loss", "w3_mag", "w4_normal", "w4_brute", "w4_scan"]:
            if col in global_history.columns:
                global_history[col] = pd.to_numeric(global_history[col], errors="coerce")

    return {
        "transport": transport,
        "model_metrics": model_metrics,
        "global_history": global_history,
        "inventory": pd.DataFrame(
            [
                {
                    "experiment": config.name,
                    "architecture": config.architecture,
                    "security": config.security,
                    "topology": config.topology,
                    "mode": config.mode,
                    "attempt_id": int(attempt_id),
                    "attempt_key": f"{config.name}#{attempt_id}",
                    "attempt_dir": str(attempt_dir),
                    "transport_files": len(transport_paths),
                    "model_metric_files": len(model_paths),
                    "global_history_files": len(global_paths),
                    "csv_files_total": len(csv_paths),
                    "pdf_files": len(list(attempt_dir.glob("*.pdf"))),
                }
            ]
        ),
    }


def load_all_data(configs: list[ExperimentConfig]) -> dict[str, pd.DataFrame]:
    transport_parts: list[pd.DataFrame] = []
    model_parts: list[pd.DataFrame] = []
    global_parts: list[pd.DataFrame] = []
    inventory_parts: list[pd.DataFrame] = []

    for config in configs:
        attempts = _discover_attempt_dirs(config)
        if not attempts:
            inventory_parts.append(
                pd.DataFrame(
                    [
                        {
                            "experiment": config.name,
                            "architecture": config.architecture,
                            "security": config.security,
                            "topology": config.topology,
                            "mode": config.mode,
                            "attempt_id": pd.NA,
                            "attempt_key": pd.NA,
                            "attempt_dir": str(config.results_root),
                            "transport_files": 0,
                            "model_metric_files": 0,
                            "global_history_files": 0,
                            "csv_files_total": 0,
                            "pdf_files": 0,
                            "missing_root": not config.results_root.exists(),
                        }
                    ]
                )
            )
            continue
        for attempt_id, attempt_dir in attempts:
            frames = _load_attempt(config, attempt_id, attempt_dir)
            if not frames["transport"].empty:
                transport_parts.append(frames["transport"])
            if not frames["model_metrics"].empty:
                model_parts.append(frames["model_metrics"])
            if not frames["global_history"].empty:
                global_parts.append(frames["global_history"])
            inventory_parts.append(frames["inventory"])

    return {
        "attempt_inventory": pd.concat(inventory_parts, ignore_index=True, sort=False) if inventory_parts else pd.DataFrame(),
        "transport": pd.concat(transport_parts, ignore_index=True, sort=False) if transport_parts else pd.DataFrame(),
        "model_metrics": pd.concat(model_parts, ignore_index=True, sort=False) if model_parts else pd.DataFrame(),
        "global_history": pd.concat(global_parts, ignore_index=True, sort=False) if global_parts else pd.DataFrame(),
    }


def build_canonical_log_events(transport: pd.DataFrame, model_metrics: pd.DataFrame, global_history: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "timestamp_dt",
        "event_name",
        "experiment",
        "architecture",
        "security",
        "topology",
        "mode",
        "attempt_id",
        "attempt_key",
        "round_ref",
        "node_id",
        "gateway_id",
        "source_scope",
        "channel",
        "operation",
        "elapsed_ms",
        "payload_plain_bytes",
        "payload_wire_bytes",
        "overhead_bytes",
        "overhead_pct",
        "expansion_ratio",
        "stage",
        "num_samples",
        "accuracy",
        "loss",
        "buffer_target",
        "client_id",
        "sample_label",
        "sample_label_name",
        "source_file",
    ]
    frames = []
    for frame in [transport, model_metrics, global_history]:
        if frame.empty:
            continue
        selected = frame.copy()
        for col in cols:
            if col not in selected.columns:
                selected[col] = pd.NA
        frames.append(selected[cols])
    if not frames:
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.sort_values(["experiment", "attempt_id", "timestamp_dt", "event_name"], na_position="last")
    return out.reset_index(drop=True)


def summarize_transport(transport: pd.DataFrame) -> pd.DataFrame:
    if transport.empty:
        return pd.DataFrame()
    frame = transport.copy()
    keys = ["experiment", "architecture", "security", "topology", "mode", "source_scope", "device_suffix", "channel", "operation"]
    for key in keys:
        if key not in frame.columns:
            frame[key] = "unknown"
    summary = (
        frame.groupby(keys, dropna=False)
        .agg(
            events=("elapsed_ms", "count"),
            avg_ms=("elapsed_ms", "mean"),
            p50_ms=("elapsed_ms", lambda s: _safe_pct(s, 50)),
            p95_ms=("elapsed_ms", lambda s: _safe_pct(s, 95)),
            p99_ms=("elapsed_ms", lambda s: _safe_pct(s, 99)),
            max_ms=("elapsed_ms", "max"),
            std_ms=("elapsed_ms", lambda s: _safe_std(s, ddof=0)),
            avg_payload_plain_bytes=("payload_plain_bytes", "mean"),
            avg_payload_wire_bytes=("payload_wire_bytes", "mean"),
            avg_overhead_bytes=("overhead_bytes", "mean"),
            avg_overhead_pct=("overhead_pct", "mean"),
            avg_expansion_ratio=("expansion_ratio", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values(["experiment", "source_scope", "channel", "operation", "device_suffix"]).reset_index(drop=True)


def summarize_model_metrics(model_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if model_metrics.empty:
        return pd.DataFrame(), pd.DataFrame()

    frame = model_metrics.copy()
    for col in ["accuracy", "loss", "num_samples", "buffer_target", "fl_round"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["stage"] = frame.get("stage", "").astype(str)

    keys = ["experiment", "architecture", "security", "topology", "mode", "device_suffix", "stage"]
    summary = (
        frame.groupby(keys, dropna=False)
        .agg(
            rows=("stage", "count"),
            rounds_seen=("fl_round", "nunique"),
            avg_num_samples=("num_samples", "mean"),
            avg_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", lambda s: _safe_std(s, ddof=0)),
            min_accuracy=("accuracy", "min"),
            max_accuracy=("accuracy", "max"),
            avg_loss=("loss", "mean"),
            std_loss=("loss", lambda s: _safe_std(s, ddof=0)),
            min_loss=("loss", "min"),
            max_loss=("loss", "max"),
            avg_buffer_target=("buffer_target", "mean"),
        )
        .reset_index()
        .sort_values(keys)
        .reset_index(drop=True)
    )

    local = summary.loc[summary["stage"].eq("local_train")].reset_index(drop=True)
    fog = summary.loc[summary["stage"].eq("fog_fedavg")].reset_index(drop=True)
    return local, fog


def enrich_global_history(global_history: pd.DataFrame) -> pd.DataFrame:
    if global_history.empty:
        return pd.DataFrame()
    frame = global_history.copy()
    for col in ["round", "accuracy", "loss", "w3_mag", "w4_normal", "w4_brute", "w4_scan"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.sort_values(["experiment", "attempt_id", "round"]).reset_index(drop=True)
    frame["round_duration_sec"] = frame.groupby(["experiment", "attempt_id"])["timestamp_dt"].diff().dt.total_seconds()
    for col in ["w3_mag", "w4_normal", "w4_brute", "w4_scan"]:
        if col in frame.columns:
            frame[f"{col}_drift"] = frame.groupby(["experiment", "attempt_id"])[col].diff().abs()
    return frame


def summarize_global_rounds(global_enriched: pd.DataFrame) -> pd.DataFrame:
    if global_enriched.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["experiment", "architecture", "security", "topology", "mode", "attempt_id", "attempt_key"]
    for keys, group in global_enriched.groupby(group_cols, dropna=False):
        group = group.sort_values("round")
        last = group.iloc[-1]
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "rounds_observed": int(group["round"].nunique()),
                "first_round": int(group["round"].min()) if group["round"].notna().any() else None,
                "last_round": int(group["round"].max()) if group["round"].notna().any() else None,
                "round_completion_rate": float(group["round"].nunique() / EXPECTED_ROUNDS_PER_ATTEMPT),
                "last_global_accuracy": _safe_float(last.get("accuracy")),
                "best_global_accuracy": _safe_float(group["accuracy"].max()),
                "avg_global_accuracy": _safe_mean(group["accuracy"]),
                "last_global_loss": _safe_float(last.get("loss")),
                "min_global_loss": _safe_float(group["loss"].min()),
                "avg_global_loss": _safe_mean(group["loss"]),
                "avg_round_duration_sec": _safe_mean(group["round_duration_sec"]),
                "p95_round_duration_sec": _safe_pct(group["round_duration_sec"], 95),
                "max_round_duration_sec": _safe_float(pd.to_numeric(group["round_duration_sec"], errors="coerce").max()),
            }
        )
        for col in ["w3_mag_drift", "w4_normal_drift", "w4_brute_drift", "w4_scan_drift"]:
            if col in group.columns:
                row[f"avg_{col}"] = _safe_mean(group[col])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def summarize_gateway_alignment(model_metrics: pd.DataFrame) -> pd.DataFrame:
    if model_metrics.empty:
        return pd.DataFrame()
    frame = model_metrics.loc[model_metrics.get("stage", "").astype(str).eq("local_train")].copy()
    if frame.empty:
        return pd.DataFrame()
    for col in ["accuracy", "loss", "fl_round"]:
        frame[col] = pd.to_numeric(frame.get(col), errors="coerce")

    rows = []
    group_cols = ["experiment", "architecture", "security", "topology", "mode", "attempt_id", "attempt_key", "fl_round"]
    for keys, group in frame.groupby(group_cols, dropna=False):
        ts = pd.to_datetime(group["timestamp_dt"], errors="coerce")
        if ts.notna().sum() >= 2:
            ts_skew = (ts.max() - ts.min()).total_seconds()
        else:
            ts_skew = float("nan")
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "round_ref": _safe_float(row["fl_round"]) + 1 if not pd.isna(row["fl_round"]) else float("nan"),
                "gateway_count": int(group["device_suffix"].nunique()),
                "local_accuracy_min": _safe_float(group["accuracy"].min()),
                "local_accuracy_max": _safe_float(group["accuracy"].max()),
                "local_accuracy_skew": _safe_float(group["accuracy"].max() - group["accuracy"].min()),
                "local_loss_min": _safe_float(group["loss"].min()),
                "local_loss_max": _safe_float(group["loss"].max()),
                "local_loss_skew": _safe_float(group["loss"].max() - group["loss"].min()),
                "gateway_sync_skew_sec": ts_skew,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def summarize_class_mix(transport: pd.DataFrame) -> pd.DataFrame:
    if transport.empty:
        return pd.DataFrame()
    if "sample_label_name" not in transport.columns and "sample_label" not in transport.columns:
        return pd.DataFrame()

    frame = transport.copy()
    label_name = frame.get("sample_label_name", pd.Series([pd.NA] * len(frame))).astype("string")
    label_fallback = frame.get("sample_label", pd.Series([pd.NA] * len(frame))).astype("string")
    frame["class_name"] = label_name.fillna(label_fallback).fillna("unknown")
    frame = frame.loc[frame["class_name"].notna() & frame["class_name"].ne("") & frame["source_scope"].eq("gateway_transport")]
    if frame.empty:
        return pd.DataFrame()

    keys = ["experiment", "architecture", "security", "topology", "mode", "attempt_id", "device_suffix", "class_name"]
    counts = frame.groupby(keys, dropna=False).size().reset_index(name="samples_seen")
    totals = counts.groupby(["experiment", "attempt_id", "device_suffix"], dropna=False)["samples_seen"].transform("sum")
    counts["pct_within_gateway_attempt"] = counts["samples_seen"] / totals * 100.0
    return counts.sort_values(keys).reset_index(drop=True)


def build_round_trace_summary(
    global_enriched: pd.DataFrame,
    model_metrics: pd.DataFrame,
    transport: pd.DataFrame,
) -> pd.DataFrame:
    if global_enriched.empty:
        return pd.DataFrame()

    base_cols = ["experiment", "architecture", "security", "topology", "mode", "attempt_id", "attempt_key", "round"]
    traces = global_enriched[base_cols + ["accuracy", "loss", "round_duration_sec"]].copy()
    traces = traces.rename(columns={"accuracy": "global_accuracy", "loss": "global_loss", "round": "round_ref"})
    traces["round_ref"] = pd.to_numeric(traces["round_ref"], errors="coerce")

    if not model_metrics.empty:
        models = model_metrics.copy()
        models["round_ref"] = pd.to_numeric(models.get("round_ref"), errors="coerce")
        for col in ["accuracy", "loss", "num_samples"]:
            models[col] = pd.to_numeric(models.get(col), errors="coerce")
        local = models.loc[models.get("stage", "").astype(str).eq("local_train")]
        if not local.empty:
            local_agg = (
                local.groupby(["experiment", "attempt_id", "round_ref"], dropna=False)
                .agg(
                    local_gateway_count=("device_suffix", "nunique"),
                    local_train_events=("stage", "count"),
                    local_accuracy_avg=("accuracy", "mean"),
                    local_loss_avg=("loss", "mean"),
                    local_samples_total=("num_samples", "sum"),
                )
                .reset_index()
            )
            traces = traces.merge(local_agg, on=["experiment", "attempt_id", "round_ref"], how="left")

        fog = models.loc[models.get("stage", "").astype(str).eq("fog_fedavg")]
        if not fog.empty:
            fog_agg = (
                fog.groupby(["experiment", "attempt_id", "round_ref"], dropna=False)
                .agg(
                    fog_fedavg_events=("stage", "count"),
                    fog_accuracy_avg=("accuracy", "mean"),
                    fog_loss_avg=("loss", "mean"),
                )
                .reset_index()
            )
            traces = traces.merge(fog_agg, on=["experiment", "attempt_id", "round_ref"], how="left")

    if not transport.empty:
        trans = transport.copy()
        trans["round_ref"] = pd.to_numeric(trans.get("round_ref"), errors="coerce")

        def _filtered_agg(mask: pd.Series, prefix: str) -> pd.DataFrame:
            subset = trans.loc[mask].copy()
            if subset.empty:
                return pd.DataFrame(columns=["experiment", "attempt_id", "round_ref"])
            out = (
                subset.groupby(["experiment", "attempt_id", "round_ref"], dropna=False)
                .agg(
                    **{
                        f"{prefix}_events": ("elapsed_ms", "count"),
                        f"{prefix}_p95_ms": ("elapsed_ms", lambda s: _safe_pct(s, 95)),
                        f"{prefix}_avg_wire_bytes": ("payload_wire_bytes", "mean"),
                    }
                )
                .reset_index()
            )
            return out

        edge = _filtered_agg(
            trans["source_scope"].eq("gateway_transport") & trans["channel"].eq("ESP32->RPi"),
            "edge_rpi",
        )
        ingress = _filtered_agg(
            trans["source_scope"].eq("server_transport") & trans["channel"].eq("RPi->PC"),
            "rpi_pc",
        )
        egress = _filtered_agg(
            trans["source_scope"].eq("server_transport") & trans["channel"].eq("PC->RPi"),
            "pc_rpi",
        )
        total = (
            trans.groupby(["experiment", "attempt_id", "round_ref"], dropna=False)
            .agg(total_transport_events=("elapsed_ms", "count"), total_transport_p95_ms=("elapsed_ms", lambda s: _safe_pct(s, 95)))
            .reset_index()
        )
        for agg in [edge, ingress, egress, total]:
            if not agg.empty:
                traces = traces.merge(agg, on=["experiment", "attempt_id", "round_ref"], how="left")

    return traces.sort_values(["experiment", "attempt_id", "round_ref"]).reset_index(drop=True)


def build_attempt_level_metrics(
    transport: pd.DataFrame,
    global_summary: pd.DataFrame,
    model_metrics: pd.DataFrame,
) -> pd.DataFrame:
    inventory_keys = ["experiment", "architecture", "security", "topology", "mode", "attempt_id", "attempt_key"]
    rows = []
    keys_seen: set[tuple[Any, ...]] = set()
    for frame in [transport, global_summary, model_metrics]:
        if frame.empty:
            continue
        for values in frame[inventory_keys].drop_duplicates().itertuples(index=False, name=None):
            keys_seen.add(values)

    for values in sorted(keys_seen):
        row = dict(zip(inventory_keys, values))
        exp = row["experiment"]
        attempt_id = row["attempt_id"]
        trans = transport.loc[(transport["experiment"].eq(exp)) & (transport["attempt_id"].eq(attempt_id))] if not transport.empty else pd.DataFrame()
        glob = global_summary.loc[(global_summary["experiment"].eq(exp)) & (global_summary["attempt_id"].eq(attempt_id))] if not global_summary.empty else pd.DataFrame()
        models = model_metrics.loc[(model_metrics["experiment"].eq(exp)) & (model_metrics["attempt_id"].eq(attempt_id))] if not model_metrics.empty else pd.DataFrame()

        if not trans.empty:
            edge = trans.loc[trans["source_scope"].eq("gateway_transport") & trans["channel"].eq("ESP32->RPi")]
            ingress = trans.loc[trans["source_scope"].eq("server_transport") & trans["channel"].eq("RPi->PC")]
            egress = trans.loc[trans["source_scope"].eq("server_transport") & trans["channel"].eq("PC->RPi")]
            row.update(
                {
                    "total_transport_events": int(trans["elapsed_ms"].count()),
                    "total_ascon_ops": int(trans["operation"].isin(["encrypt", "decrypt"]).sum()),
                    "edge_processing_p95_ms": _safe_pct(edge["elapsed_ms"], 95),
                    "edge_payload_wire_bytes_avg": _safe_mean(edge["payload_wire_bytes"]),
                    "server_ingress_processing_p95_ms": _safe_pct(ingress["elapsed_ms"], 95),
                    "server_egress_processing_p95_ms": _safe_pct(egress["elapsed_ms"], 95),
                    "avg_transport_overhead_pct": _safe_mean(trans["overhead_pct"]),
                }
            )
        if not glob.empty:
            first = glob.iloc[0]
            row.update(
                {
                    "rounds_observed": _safe_float(first.get("rounds_observed")),
                    "round_completion_rate": _safe_float(first.get("round_completion_rate")),
                    "last_global_accuracy": _safe_float(first.get("last_global_accuracy")),
                    "last_global_loss": _safe_float(first.get("last_global_loss")),
                    "avg_round_duration_sec": _safe_float(first.get("avg_round_duration_sec")),
                    "p95_round_duration_sec": _safe_float(first.get("p95_round_duration_sec")),
                }
            )
        if not models.empty:
            local = models.loc[models.get("stage", "").astype(str).eq("local_train")]
            row.update(
                {
                    "local_accuracy_avg": _safe_mean(local.get("accuracy", pd.Series(dtype=float))),
                    "local_loss_avg": _safe_mean(local.get("loss", pd.Series(dtype=float))),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["architecture", "topology", "security", "experiment", "attempt_id"]).reset_index(drop=True)


def build_nist_tables(transport: pd.DataFrame, attempt_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if transport.empty:
        ascon_attempts = pd.DataFrame()
    else:
        ascon = transport.loc[transport["security"].eq("ASCON")].copy()
        if ascon.empty:
            ascon_attempts = pd.DataFrame()
        else:
            ascon_attempts = (
                ascon.groupby(["experiment", "architecture", "topology", "attempt_id", "attempt_key"], dropna=False)
                .agg(
                    ascon_ops=("operation", lambda s: int(s.isin(["encrypt", "decrypt"]).sum())),
                    encrypt_ops=("operation", lambda s: int(s.eq("encrypt").sum())),
                    decrypt_ops=("operation", lambda s: int(s.eq("decrypt").sum())),
                    p95_latency_ms=("elapsed_ms", lambda s: _safe_pct(s, 95)),
                    max_latency_ms=("elapsed_ms", "max"),
                    avg_overhead_pct=("overhead_pct", "mean"),
                    avg_overhead_bytes=("overhead_bytes", "mean"),
                    avg_expansion_ratio=("expansion_ratio", "mean"),
                )
                .reset_index()
            )
            ascon_attempts["passes_project_min_1000_ops"] = ascon_attempts["ascon_ops"] >= PROJECT_MIN_ASCON_OPS
            ascon_attempts["passes_project_p95_50ms"] = ascon_attempts["p95_latency_ms"] <= PROJECT_LATENCY_P95_LIMIT_MS
            ascon_attempts["passes_project_overhead_50pct"] = ascon_attempts["avg_overhead_pct"] <= PROJECT_OVERHEAD_LIMIT_PCT

    if ascon_attempts.empty:
        ascon_experiment = pd.DataFrame()
    else:
        ascon_experiment = (
            ascon_attempts.groupby(["experiment", "architecture", "topology"], dropna=False)
            .agg(
                attempts=("attempt_id", "nunique"),
                total_ascon_ops=("ascon_ops", "sum"),
                avg_ascon_ops_per_attempt=("ascon_ops", "mean"),
                attempts_ge_1000_ops=("passes_project_min_1000_ops", "sum"),
                attempts_p95_le_50ms=("passes_project_p95_50ms", "sum"),
                attempts_overhead_le_50pct=("passes_project_overhead_50pct", "sum"),
                p95_latency_ms_avg=("p95_latency_ms", "mean"),
                p95_latency_ms_max=("p95_latency_ms", "max"),
                avg_overhead_pct=("avg_overhead_pct", "mean"),
                avg_expansion_ratio=("avg_expansion_ratio", "mean"),
            )
            .reset_index()
        )
        ascon_experiment["attempts_ge_30_independent_runs"] = ascon_experiment["attempts"] >= RECOMMENDED_INDEPENDENT_ATTEMPTS

    audit = pd.DataFrame(
        [
            {
                "claim": "ASCON is relevant because NIST selected the Ascon family for lightweight cryptography standardization.",
                "source": "NIST Lightweight Cryptography standardization announcements and final standard page.",
                "project_evidence": "CSV transport rows marked ASCON show the project executed encrypt/decrypt operations on ESP32/RPi/PC paths.",
                "status": "supported",
                "caution": "Do not claim NIST certified this specific implementation; NIST standardized the algorithm family.",
            },
            {
                "claim": ">=1000 ASCON operations is enough evidence for a project benchmark.",
                "source": "Project benchmark criterion.",
                "project_evidence": "nist_ascon_attempt_summary.csv counts ASCON operations per attempt.",
                "status": "project_threshold",
                "caution": "This is not an official NIST LWC pass/fail requirement.",
            },
            {
                "claim": "p95 latency <= 50 ms is acceptable for the project IoT path.",
                "source": "Project SRE/SLO criterion.",
                "project_evidence": "transport_sli_summary.csv and nist_ascon_attempt_summary.csv compute p95 elapsed_ms.",
                "status": "project_threshold",
                "caution": "This is a system SLO, not a NIST cryptographic requirement.",
            },
            {
                "claim": "Average overhead <= 50% is a project efficiency objective.",
                "source": "Project efficiency criterion.",
                "project_evidence": "CSV fields pt_bytes, enc_bytes and overhead_bytes compute expansion and overhead.",
                "status": "project_threshold",
                "caution": "AEAD overhead percentage grows when plaintext payloads are small.",
            },
            {
                "claim": "NIST SP 800-22 justifies t-test/Wilcoxon for FL model metrics.",
                "source": "NIST SP 800-22 Statistical Test Suite for random and pseudorandom number generators.",
                "project_evidence": "Not applicable to FL model significance tests.",
                "status": "not_supported",
                "caution": "Use standard statistical methodology for model comparisons; cite SP 800-22 only for RNG/randomness context.",
            },
            {
                "claim": "n >= 30 independent executions is desirable for strong statistical inference.",
                "source": "General experimental design rule of thumb, not SP 800-22.",
                "project_evidence": "attempt_inventory.csv counts independent attempt folders per experiment.",
                "status": "recommended_but_not_met_per_variant",
                "caution": "Rounds inside the same execution are repeated measures, not independent executions.",
            },
        ]
    )
    return ascon_attempts, ascon_experiment, audit


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    greater = 0
    lower = 0
    for x in a:
        greater += int(np.sum(x > b))
        lower += int(np.sum(x < b))
    return float((greater - lower) / (len(a) * len(b)))


def _choose_stat_test(a: pd.Series, b: pd.Series) -> dict[str, Any]:
    arr_a = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
    arr_b = pd.to_numeric(b, errors="coerce").dropna().to_numpy(dtype=float)
    result: dict[str, Any] = {
        "n_ascon": int(len(arr_a)),
        "n_no_ascon": int(len(arr_b)),
        "mean_ascon": float(np.mean(arr_a)) if len(arr_a) else float("nan"),
        "mean_no_ascon": float(np.mean(arr_b)) if len(arr_b) else float("nan"),
        "median_ascon": float(np.median(arr_a)) if len(arr_a) else float("nan"),
        "median_no_ascon": float(np.median(arr_b)) if len(arr_b) else float("nan"),
        "mean_diff_ascon_minus_no_ascon": float(np.mean(arr_a) - np.mean(arr_b)) if len(arr_a) and len(arr_b) else float("nan"),
        "cliffs_delta": _cliffs_delta(arr_a, arr_b),
        "test": "insufficient_data",
        "p_value": float("nan"),
        "significant_alpha_0_05": False,
        "normality_note": "not_tested",
    }
    if len(arr_a) < 2 or len(arr_b) < 2 or stats is None:
        if stats is None:
            result["test"] = "not_available_scipy_missing"
        return result

    normal_a = False
    normal_b = False
    if len(arr_a) >= 3 and len(np.unique(arr_a)) > 1:
        normal_a = bool(stats.shapiro(arr_a).pvalue >= ALPHA)
    if len(arr_b) >= 3 and len(np.unique(arr_b)) > 1:
        normal_b = bool(stats.shapiro(arr_b).pvalue >= ALPHA)

    if normal_a and normal_b:
        stat = stats.ttest_ind(arr_a, arr_b, equal_var=False, nan_policy="omit")
        result["test"] = "welch_t_test_independent"
        result["p_value"] = float(stat.pvalue)
        result["normality_note"] = "both_groups_shapiro_not_rejected"
    else:
        stat = stats.mannwhitneyu(arr_a, arr_b, alternative="two-sided")
        result["test"] = "mann_whitney_u_independent"
        result["p_value"] = float(stat.pvalue)
        result["normality_note"] = "non_normal_or_small_sample"

    result["significant_alpha_0_05"] = bool(result["p_value"] < ALPHA)
    return result


def build_statistical_comparisons(attempt_metrics: pd.DataFrame) -> pd.DataFrame:
    if attempt_metrics.empty:
        return pd.DataFrame()
    metrics = [
        "edge_processing_p95_ms",
        "edge_payload_wire_bytes_avg",
        "server_ingress_processing_p95_ms",
        "server_egress_processing_p95_ms",
        "avg_transport_overhead_pct",
        "avg_round_duration_sec",
        "p95_round_duration_sec",
        "last_global_accuracy",
        "last_global_loss",
        "round_completion_rate",
        "local_accuracy_avg",
        "local_loss_avg",
    ]
    rows = []
    for (architecture, topology), group in attempt_metrics.groupby(["architecture", "topology"], dropna=False):
        ascon = group.loc[group["security"].eq("ASCON")]
        plain = group.loc[group["security"].eq("NO_ASCON")]
        if ascon.empty or plain.empty:
            continue
        for metric in metrics:
            if metric not in group.columns:
                continue
            test = _choose_stat_test(ascon[metric], plain[metric])
            test.update(
                {
                    "architecture": architecture,
                    "topology": topology,
                    "metric": metric,
                    "comparison": "ASCON_vs_NO_ASCON",
                    "interpretation": _interpret_stat_metric(metric, test),
                }
            )
            rows.append(test)
    return pd.DataFrame(rows).sort_values(["architecture", "topology", "metric"]).reset_index(drop=True)


def _interpret_stat_metric(metric: str, test: dict[str, Any]) -> str:
    if not test.get("significant_alpha_0_05"):
        return "No hay diferencia estadisticamente significativa con alpha=0.05 en estos intentos."
    diff = test.get("mean_diff_ascon_minus_no_ascon", float("nan"))
    if metric.endswith("_ms") or "duration" in metric:
        direction = "ASCON mayor latencia/tiempo" if diff > 0 else "ASCON menor latencia/tiempo"
    elif "payload" in metric or "overhead" in metric:
        direction = "ASCON mayor carga de bytes" if diff > 0 else "ASCON menor carga de bytes"
    elif "accuracy" in metric:
        direction = "ASCON mayor accuracy" if diff > 0 else "ASCON menor accuracy"
    elif "loss" in metric:
        direction = "ASCON mayor loss" if diff > 0 else "ASCON menor loss"
    else:
        direction = "diferencia significativa"
    return f"Diferencia significativa: {direction}."


def build_catalogs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_catalog = pd.DataFrame(
        [
            {"metric": "edge_rpi_processing_p95_ms", "source": "transport_sli_summary.csv", "meaning": "p95 de decrypt/deserialize ESP32->RPi por gateway.", "sli_type": "latency"},
            {"metric": "rpi_pc_processing_p95_ms", "source": "transport_sli_summary.csv", "meaning": "p95 de decrypt/deserialize RPi->PC en servidor.", "sli_type": "latency"},
            {"metric": "pc_rpi_processing_p95_ms", "source": "transport_sli_summary.csv", "meaning": "p95 de encrypt/serialize PC->RPi.", "sli_type": "latency"},
            {"metric": "avg_payload_wire_bytes", "source": "transport_sli_summary.csv", "meaning": "Bytes transmitidos por mensaje luego de cifrado o serializacion.", "sli_type": "traffic"},
            {"metric": "avg_overhead_pct", "source": "transport_sli_summary.csv", "meaning": "Sobrecosto porcentual respecto a plaintext/payload.", "sli_type": "efficiency"},
            {"metric": "local_accuracy_avg", "source": "local_training_sli_summary.csv", "meaning": "Accuracy local promedio por gateway.", "sli_type": "model_quality"},
            {"metric": "local_loss_avg", "source": "local_training_sli_summary.csv", "meaning": "Loss local promedio por gateway.", "sli_type": "model_quality"},
            {"metric": "round_completion_rate", "source": "global_round_sli_summary.csv", "meaning": "Rondas globales observadas / 30 esperadas.", "sli_type": "availability"},
            {"metric": "gateway_sync_skew_sec", "source": "gateway_alignment_sli_summary.csv", "meaning": "Diferencia temporal entre gateways para una misma ronda local.", "sli_type": "coordination"},
            {"metric": "weight_drift", "source": "global_round_enriched.csv", "meaning": "Cambio absoluto entre rondas de magnitudes W3/W4.", "sli_type": "convergence"},
        ]
    )
    log_catalog = pd.DataFrame(
        [
            {"event_name": "transport.crypto.gateway", "source": "canonical_log_events.csv", "when": "Cada decrypt ASCON en gateway.", "required_fields": "timestamp_dt,channel,operation,elapsed_ms,pt_bytes,enc_bytes,fl_round"},
            {"event_name": "transport.crypto.server", "source": "canonical_log_events.csv", "when": "Cada decrypt/encrypt ASCON en servidor.", "required_fields": "timestamp_dt,channel,operation,elapsed_ms,pt_bytes,enc_bytes,fl_round"},
            {"event_name": "transport.plain.gateway", "source": "canonical_log_events.csv", "when": "Cada deserialize/serialize sin ASCON en gateway.", "required_fields": "timestamp_dt,channel,operation,payload_bytes,elapsed_ms,fl_round"},
            {"event_name": "transport.plain.server", "source": "canonical_log_events.csv", "when": "Cada deserialize/serialize sin ASCON en servidor.", "required_fields": "timestamp_dt,channel,operation,payload_bytes,elapsed_ms,fl_round"},
            {"event_name": "model.local_train", "source": "canonical_log_events.csv", "when": "Fin de entrenamiento local.", "required_fields": "timestamp_dt,gateway_id,fl_round,num_samples,accuracy,loss"},
            {"event_name": "model.fog_fedavg", "source": "canonical_log_events.csv", "when": "Agregacion fog en lider.", "required_fields": "timestamp_dt,gateway_id,fl_round,num_samples,accuracy,loss,peer_count"},
            {"event_name": "model.global_round", "source": "canonical_log_events.csv", "when": "Servidor registra ronda global.", "required_fields": "time,round,accuracy,loss,w3_mag,w4_*"},
        ]
    )
    trace_catalog = pd.DataFrame(
        [
            {"trace": "round_trace", "span": "edge_rpi_transport", "source": "transport gateway", "join_key": "experiment,attempt_id,round_ref"},
            {"trace": "round_trace", "span": "local_train", "source": "model_metrics stage=local_train", "join_key": "experiment,attempt_id,round_ref"},
            {"trace": "round_trace", "span": "fog_fedavg", "source": "model_metrics stage=fog_fedavg", "join_key": "experiment,attempt_id,round_ref"},
            {"trace": "round_trace", "span": "rpi_pc_transport", "source": "transport server channel=RPi->PC", "join_key": "experiment,attempt_id,round_ref"},
            {"trace": "round_trace", "span": "pc_rpi_transport", "source": "transport server channel=PC->RPi", "join_key": "experiment,attempt_id,round_ref"},
            {"trace": "round_trace", "span": "global_state", "source": "global_weights_history", "join_key": "experiment,attempt_id,round_ref"},
        ]
    )
    dashboard_catalog = pd.DataFrame(
        [
            {"panel": "Salud del sistema", "metrics": "round_completion_rate,last_global_accuracy,last_global_loss,avg_round_duration_sec", "source": "global_round_sli_summary.csv"},
            {"panel": "Transporte", "metrics": "p95_ms,avg_payload_wire_bytes,avg_overhead_pct", "source": "transport_sli_summary.csv"},
            {"panel": "Entrenamiento local", "metrics": "avg_accuracy,avg_loss,avg_num_samples", "source": "local_training_sli_summary.csv"},
            {"panel": "Agregacion fog", "metrics": "fog_fedavg accuracy/loss,peer_count", "source": "fog_aggregation_sli_summary.csv"},
            {"panel": "Convergencia global", "metrics": "global accuracy/loss,w3_mag,w4_*", "source": "global_round_enriched.csv"},
            {"panel": "Calidad de datos", "metrics": "class mix por gateway/intento", "source": "class_mix_summary.csv"},
            {"panel": "Trazas por ronda", "metrics": "round_trace_summary", "source": "round_trace_summary.csv"},
            {"panel": "NIST / ASCON", "metrics": "ascon_ops,p95_latency,overhead_pct", "source": "nist_ascon_experiment_summary.csv"},
        ]
    )
    return metric_catalog, log_catalog, trace_catalog, dashboard_catalog


def _plot_transport_latency(transport_sli: pd.DataFrame, output: Path) -> None:
    if transport_sli.empty:
        return
    frame = transport_sli.copy()
    mask = (
        (frame["source_scope"].eq("gateway_transport") & frame["channel"].eq("ESP32->RPi"))
        | (frame["source_scope"].eq("server_transport") & frame["channel"].isin(["RPi->PC", "PC->RPi"]))
    )
    frame = frame.loc[mask].copy()
    if frame.empty:
        return
    frame["path"] = frame["source_scope"].str.replace("_transport", "", regex=False) + " " + frame["channel"] + " " + frame["operation"]
    plot_df = frame.groupby(["experiment", "path"], dropna=False)["p95_ms"].mean().reset_index()
    pivot = plot_df.pivot(index="experiment", columns="path", values="p95_ms").sort_index()
    ax = pivot.plot(kind="bar", figsize=(16, 7), width=0.82)
    ax.set_title("Transport latency p95 by experiment")
    ax.set_xlabel("Experimento")
    ax.set_ylabel("p95 elapsed_ms")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def _plot_payload_bytes(transport_sli: pd.DataFrame, output: Path) -> None:
    if transport_sli.empty:
        return
    frame = transport_sli.copy()
    frame["path"] = frame["source_scope"].str.replace("_transport", "", regex=False) + " " + frame["channel"]
    pivot = frame.groupby(["experiment", "path"], dropna=False)["avg_payload_wire_bytes"].mean().reset_index().pivot(
        index="experiment", columns="path", values="avg_payload_wire_bytes"
    )
    ax = pivot.sort_index().plot(kind="bar", figsize=(16, 7), width=0.82)
    ax.set_title("Average wire payload bytes by experiment")
    ax.set_xlabel("Experimento")
    ax.set_ylabel("Bytes promedio en wire/payload")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def _plot_global_accuracy_loss(global_enriched: pd.DataFrame, output: Path) -> None:
    if global_enriched.empty:
        return
    frame = global_enriched.groupby(["experiment", "round"], dropna=False).agg(accuracy=("accuracy", "mean"), loss=("loss", "mean")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for experiment, group in frame.groupby("experiment"):
        axes[0].plot(group["round"], group["accuracy"], marker="o", linewidth=1.5, label=experiment)
        axes[1].plot(group["round"], group["loss"], marker="o", linewidth=1.5, label=experiment)
    axes[0].set_title("Global accuracy over rounds")
    axes[0].set_xlabel("Ronda")
    axes[0].set_ylabel("Accuracy")
    axes[1].set_title("Global loss over rounds")
    axes[1].set_xlabel("Ronda")
    axes[1].set_ylabel("Loss")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def _plot_weight_trends(global_enriched: pd.DataFrame, output: Path) -> None:
    if global_enriched.empty:
        return
    weight_cols = [col for col in ["w3_mag", "w4_normal", "w4_brute", "w4_scan"] if col in global_enriched.columns]
    if not weight_cols:
        return
    frame = global_enriched.groupby(["experiment", "round"], dropna=False)[weight_cols].mean().reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    for ax, col in zip(axes, weight_cols):
        for experiment, group in frame.groupby("experiment"):
            ax.plot(group["round"], group[col], linewidth=1.5, label=experiment)
        ax.set_title(col)
        ax.set_xlabel("Ronda")
        ax.set_ylabel("Magnitud")
    axes[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def _plot_round_duration(global_enriched: pd.DataFrame, output: Path) -> None:
    if global_enriched.empty or "round_duration_sec" not in global_enriched.columns:
        return
    frame = global_enriched.dropna(subset=["round_duration_sec"])
    if frame.empty:
        return
    data = [group["round_duration_sec"].dropna().to_numpy(dtype=float) for _, group in frame.groupby("experiment")]
    labels = [name for name, _ in frame.groupby("experiment")]
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_title("Round duration distribution")
    ax.set_xlabel("Experimento")
    ax.set_ylabel("Duracion de ronda (s)")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def _plot_gateway_accuracy_skew(alignment: pd.DataFrame, output: Path) -> None:
    if alignment.empty:
        return
    frame = alignment.groupby(["experiment", "round_ref"], dropna=False)["local_accuracy_skew"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(16, 7))
    for experiment, group in frame.groupby("experiment"):
        ax.plot(group["round_ref"], group["local_accuracy_skew"], marker="o", linewidth=1.5, label=experiment)
    ax.set_title("Gateway accuracy skew by round")
    ax.set_xlabel("Ronda")
    ax.set_ylabel("Max accuracy - min accuracy")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def _plot_class_mix(class_mix: pd.DataFrame, output: Path) -> None:
    if class_mix.empty:
        return
    frame = class_mix.groupby(["experiment", "device_suffix", "class_name"], dropna=False)["samples_seen"].sum().reset_index()
    frame["gateway"] = frame["experiment"] + " | " + frame["device_suffix"].astype(str)
    totals = frame.groupby("gateway")["samples_seen"].transform("sum")
    frame["pct"] = frame["samples_seen"] / totals * 100.0
    pivot = frame.pivot_table(index="gateway", columns="class_name", values="pct", aggfunc="sum").fillna(0.0)
    if len(pivot) > 30:
        aggregate = class_mix.groupby(["experiment", "class_name"], dropna=False)["samples_seen"].sum().reset_index()
        totals = aggregate.groupby("experiment")["samples_seen"].transform("sum")
        aggregate["pct"] = aggregate["samples_seen"] / totals * 100.0
        pivot = aggregate.pivot_table(index="experiment", columns="class_name", values="pct", aggfunc="sum").fillna(0.0)
    ax = pivot.sort_index().plot(kind="bar", stacked=True, figsize=(16, 8), width=0.82)
    ax.set_title("Class mix by gateway/experiment")
    ax.set_xlabel("Gateway o experimento")
    ax.set_ylabel("% muestras")
    ax.legend(title="Clase", fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def _plot_nist_ops(nist_exp: pd.DataFrame, output: Path) -> None:
    if nist_exp.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(nist_exp["experiment"], nist_exp["total_ascon_ops"])
    ax.axhline(PROJECT_MIN_ASCON_OPS, color="red", linestyle="--", label="1000 ops por intento (referencia)")
    ax.set_title("Total ASCON operations by experiment")
    ax.set_xlabel("Experimento ASCON")
    ax.set_ylabel("Operaciones encrypt/decrypt")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def _plot_nist_overhead(nist_exp: pd.DataFrame, output: Path) -> None:
    if nist_exp.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(nist_exp["experiment"], nist_exp["avg_overhead_pct"])
    ax.axhline(PROJECT_OVERHEAD_LIMIT_PCT, color="red", linestyle="--", label="50% objetivo proyecto")
    ax.set_title("Average ASCON overhead percentage")
    ax.set_xlabel("Experimento ASCON")
    ax.set_ylabel("Overhead promedio (%)")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def generate_figures(
    output_root: Path,
    transport_sli: pd.DataFrame,
    global_enriched: pd.DataFrame,
    alignment: pd.DataFrame,
    class_mix: pd.DataFrame,
    nist_exp: pd.DataFrame,
) -> None:
    _plot_transport_latency(transport_sli, output_root / "transport_latency_p95.png")
    _plot_payload_bytes(transport_sli, output_root / "transport_payload_bytes.png")
    _plot_global_accuracy_loss(global_enriched, output_root / "global_accuracy_loss.png")
    _plot_weight_trends(global_enriched, output_root / "weight_magnitude_trends.png")
    _plot_round_duration(global_enriched, output_root / "round_duration.png")
    _plot_gateway_accuracy_skew(alignment, output_root / "gateway_accuracy_skew.png")
    _plot_class_mix(class_mix, output_root / "class_mix_by_gateway.png")
    _plot_nist_ops(nist_exp, output_root / "nist_ascon_operations.png")
    _plot_nist_overhead(nist_exp, output_root / "nist_ascon_overhead_pct.png")


def build_executive_summary(
    inventory: pd.DataFrame,
    transport: pd.DataFrame,
    model_metrics: pd.DataFrame,
    global_summary: pd.DataFrame,
    nist_exp: pd.DataFrame,
    stats_summary: pd.DataFrame,
) -> dict[str, Any]:
    by_experiment: dict[str, Any] = {}
    for experiment, group in inventory.dropna(subset=["attempt_id"]).groupby("experiment"):
        global_group = global_summary.loc[global_summary["experiment"].eq(experiment)] if not global_summary.empty else pd.DataFrame()
        transport_group = transport.loc[transport["experiment"].eq(experiment)] if not transport.empty else pd.DataFrame()
        model_group = model_metrics.loc[model_metrics["experiment"].eq(experiment)] if not model_metrics.empty else pd.DataFrame()
        by_experiment[experiment] = {
            "attempts": int(group["attempt_id"].nunique()),
            "csv_files_total": int(group["csv_files_total"].sum()),
            "transport_events": int(len(transport_group)),
            "model_metric_rows": int(len(model_group)),
            "global_rounds": int(global_group["rounds_observed"].sum()) if not global_group.empty else 0,
            "avg_last_global_accuracy": _safe_mean(global_group["last_global_accuracy"]) if not global_group.empty else float("nan"),
            "avg_last_global_loss": _safe_mean(global_group["last_global_loss"]) if not global_group.empty else float("nan"),
            "avg_round_completion_rate": _safe_mean(global_group["round_completion_rate"]) if not global_group.empty else float("nan"),
        }

    nist_block: dict[str, Any]
    if nist_exp.empty:
        nist_block = {"ascon_experiments": 0}
    else:
        nist_block = {
            "ascon_experiments": int(nist_exp["experiment"].nunique()),
            "total_ascon_ops": int(nist_exp["total_ascon_ops"].sum()),
            "attempts_ge_1000_ops": int(nist_exp["attempts_ge_1000_ops"].sum()),
            "max_p95_latency_ms": _safe_float(nist_exp["p95_latency_ms_max"].max()),
            "avg_overhead_pct_mean": _safe_mean(nist_exp["avg_overhead_pct"]),
            "note": "Umbrales 1000 ops, 50 ms y 50% son criterios operativos del proyecto; no son pass/fail oficiales NIST.",
        }

    significant = []
    if not stats_summary.empty:
        sig = stats_summary.loc[stats_summary["significant_alpha_0_05"].eq(True)].copy()
        for _, row in sig.iterrows():
            significant.append(
                {
                    "architecture": row.get("architecture"),
                    "topology": row.get("topology"),
                    "metric": row.get("metric"),
                    "test": row.get("test"),
                    "p_value": _safe_float(row.get("p_value")),
                    "interpretation": row.get("interpretation"),
                }
            )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "output_root": str(OUTPUT_ROOT),
        "totals": {
            "experiments_configured": len(EXPERIMENTS),
            "attempts_discovered": int(inventory.dropna(subset=["attempt_id"]).shape[0]) if "attempt_id" in inventory.columns else 0,
            "attempt_ids_unique_across_all_experiments": int(inventory["attempt_id"].dropna().nunique()) if "attempt_id" in inventory.columns else 0,
            "experiment_attempt_pairs": int(inventory.dropna(subset=["attempt_id"]).shape[0]) if "attempt_id" in inventory.columns else 0,
            "csv_files_total": int(inventory["csv_files_total"].sum()) if "csv_files_total" in inventory.columns else 0,
            "transport_events": int(len(transport)),
            "model_metric_rows": int(len(model_metrics)),
            "global_attempt_summaries": int(len(global_summary)),
        },
        "by_experiment": by_experiment,
        "nist": nist_block,
        "statistical_highlights_alpha_0_05": significant,
        "limitations": [
            "La latencia extremo a extremo real por muestra no se puede reconstruir exacta sin trace_id compartido.",
            "Las filas del servidor no siempre identifican gateway de origen; se analizan como transporte agregado.",
            "Las rondas dentro de una ejecucion son medidas repetidas, no ejecuciones independientes.",
            "NIST SP 800-22 aplica a pruebas de aleatoriedad RNG/PRNG; no debe citarse como base directa de t-test o Wilcoxon para FL.",
        ],
    }


def write_sre_spec(output_root: Path) -> None:
    text = f"""# SRE Observability Spec - HFL v7 Completo

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
- `round_completion_rate`: rondas observadas / {EXPECTED_ROUNDS_PER_ATTEMPT}.
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
"""
    (output_root / "SRE_OBSERVABILITY_SPEC.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    data = load_all_data(EXPERIMENTS)
    inventory = data["attempt_inventory"]
    transport = data["transport"]
    model_metrics = data["model_metrics"]
    global_history = data["global_history"]

    canonical = build_canonical_log_events(transport, model_metrics, global_history)
    transport_sli = summarize_transport(transport)
    local_sli, fog_sli = summarize_model_metrics(model_metrics)
    global_enriched = enrich_global_history(global_history)
    global_sli = summarize_global_rounds(global_enriched)
    alignment = summarize_gateway_alignment(model_metrics)
    class_mix = summarize_class_mix(transport)
    round_trace = build_round_trace_summary(global_enriched, model_metrics, transport)
    attempt_metrics = build_attempt_level_metrics(transport, global_sli, model_metrics)
    nist_attempt, nist_experiment, nist_audit = build_nist_tables(transport, attempt_metrics)
    stats_summary = build_statistical_comparisons(attempt_metrics)
    metric_catalog, log_catalog, trace_catalog, dashboard_catalog = build_catalogs()
    executive_summary = build_executive_summary(inventory, transport, model_metrics, global_sli, nist_experiment, stats_summary)

    outputs = {
        "attempt_inventory.csv": inventory,
        "canonical_log_events.csv": canonical,
        "transport_sli_summary.csv": transport_sli,
        "local_training_sli_summary.csv": local_sli,
        "fog_aggregation_sli_summary.csv": fog_sli,
        "global_round_sli_summary.csv": global_sli,
        "global_round_enriched.csv": global_enriched,
        "gateway_alignment_sli_summary.csv": alignment,
        "class_mix_summary.csv": class_mix,
        "round_trace_summary.csv": round_trace,
        "attempt_level_metrics.csv": attempt_metrics,
        "nist_ascon_attempt_summary.csv": nist_attempt,
        "nist_ascon_experiment_summary.csv": nist_experiment,
        "nist_claim_audit.csv": nist_audit,
        "statistical_comparison_summary.csv": stats_summary,
        "metric_catalog.csv": metric_catalog,
        "log_catalog.csv": log_catalog,
        "trace_catalog.csv": trace_catalog,
        "dashboard_catalog.csv": dashboard_catalog,
    }

    for filename, frame in outputs.items():
        _write_csv(frame, OUTPUT_ROOT / filename)

    with (OUTPUT_ROOT / "executive_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(executive_summary), handle, indent=2, ensure_ascii=False)

    write_sre_spec(OUTPUT_ROOT)
    generate_figures(OUTPUT_ROOT, transport_sli, global_enriched, alignment, class_mix, nist_experiment)

    print("Analisis completo generado.")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Experimentos: {len(EXPERIMENTS)}")
    print(f"Intentos experimentales: {len(inventory.dropna(subset=['attempt_id'])) if not inventory.empty else 0}")
    print(f"Eventos transporte: {len(transport)}")
    print(f"Filas modelo: {len(model_metrics)}")
    print(f"Rondas globales: {len(global_history)}")


if __name__ == "__main__":
    main()
