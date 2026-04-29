"""
SRE consolidado para todas las variantes hfl_v7.

Este script carga los CSV historicos de las 8 carpetas de resultados sin
modificarlos y produce un analisis SRE unificado con un nuevo eje `variant`
que distingue cada combinacion de modelo (RN/CNN), seguridad (ASCON/PLAIN)
y topologia (NoFOG/FOG).

Salidas (en `Analisis de Modelos/Completo/`):
- executive_summary.json
- SRE_OBSERVABILITY_SPEC.md
- canonical_log_events.csv
- transport_sli_summary.csv
- local_training_sli_summary.csv
- global_round_sli_summary.csv
- round_trace_summary.csv
- transport_latency_p95.png
- transport_payload_bytes.png
- global_accuracy_loss.png
- weight_magnitude_trends.png
- round_duration.png
- gateway_accuracy_skew.png
- class_mix_by_gateway.png
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-darkgrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
warnings.filterwarnings("ignore")

ATTEMPT_RE = re.compile(r"#(\d+)$")

# ---------------------------------------------------------------------------
# Configuracion de variantes
# ---------------------------------------------------------------------------

REPO_SRC = Path(
    r"C:\Users\VivoBook\Downloads\microproyecto2\microproyecto 2 porque wtf\src"
)
OUTPUT_DIR = REPO_SRC / "Analisis de Modelos" / "Completo"


@dataclass(frozen=True)
class VariantSpec:
    label: str
    model: str           # RN | CNN
    security: str        # ASCON | PLAIN
    topology: str        # NoFOG | FOG
    results_dir: Path
    gateway_transport_glob: str
    server_transport_glob: str
    model_metrics_glob: str
    global_history_glob: str = "global_weights_history_*.csv"


VARIANTS: list[VariantSpec] = [
    VariantSpec(
        label="RN_ASCON_NoFOG",
        model="RN", security="ASCON", topology="NoFOG",
        results_dir=REPO_SRC / "hfl_v7-RN" / "Results",
        gateway_transport_glob="ascon_metrics_gateway_*.csv",
        server_transport_glob="ascon_metrics_server_*.csv",
        model_metrics_glob="model_metrics_gateway_*.csv",
    ),
    VariantSpec(
        label="RN_ASCON_FOG",
        model="RN", security="ASCON", topology="FOG",
        results_dir=REPO_SRC / "hfl_v7-RN" / "Results_FOG",
        gateway_transport_glob="ascon_metrics_RN_FOG_gateway_*.csv",
        server_transport_glob="ascon_metrics_server_fog_*.csv",
        model_metrics_glob="RN_FOG_model_metrics_gateway_*.csv",
    ),
    VariantSpec(
        label="RN_PLAIN_NoFOG",
        model="RN", security="PLAIN", topology="NoFOG",
        results_dir=REPO_SRC / "hfl_v7-no-ascon-RN" / "Results",
        gateway_transport_glob="plain_metrics_gateway_*.csv",
        server_transport_glob="plain_metrics_server_*.csv",
        model_metrics_glob="model_metrics_gateway_*.csv",
    ),
    VariantSpec(
        label="RN_PLAIN_FOG",
        model="RN", security="PLAIN", topology="FOG",
        results_dir=REPO_SRC / "hfl_v7-no-ascon-RN" / "Results_FOG",
        gateway_transport_glob="RN-Fog-plain_metrics_gateway_*.csv",
        server_transport_glob="plain_metrics_server_fog_*.csv",
        model_metrics_glob="RN_FOG_no_ascon_model_metrics_gateway_*.csv",
    ),
    VariantSpec(
        label="CNN_ASCON_NoFOG",
        model="CNN", security="ASCON", topology="NoFOG",
        results_dir=REPO_SRC / "hfl_v7-CNN" / "Results",
        gateway_transport_glob="ascon_metrics_CNN_gateway_*.csv",
        server_transport_glob="ascon_metrics_server_*.csv",
        model_metrics_glob="model_metrics_gateway_*.csv",
    ),
    VariantSpec(
        label="CNN_ASCON_FOG",
        model="CNN", security="ASCON", topology="FOG",
        results_dir=REPO_SRC / "hfl_v7-CNN" / "Results_FOG",
        gateway_transport_glob="ascon_metrics_CNN_FOG_gateway_*.csv",
        server_transport_glob="ascon_metrics_server_fog_*.csv",
        model_metrics_glob="model_metrics_gateway_*.csv",
    ),
    VariantSpec(
        label="CNN_PLAIN_NoFOG",
        model="CNN", security="PLAIN", topology="NoFOG",
        results_dir=REPO_SRC / "hfl_v7-no-ascon-CNN" / "Results",
        gateway_transport_glob="CNN-plain_metrics_gateway_*.csv",
        server_transport_glob="plain_metrics_server_*.csv",
        model_metrics_glob="CNN_no_ascon_model_metrics_gateway_*.csv",
    ),
    VariantSpec(
        label="CNN_PLAIN_FOG",
        model="CNN", security="PLAIN", topology="FOG",
        results_dir=REPO_SRC / "hfl_v7-no-ascon-CNN" / "Results_FOG",
        gateway_transport_glob="CNN-fog-plain_metrics_gateway_*.csv",
        server_transport_glob="plain_metrics_server_fog_*.csv",
        model_metrics_glob="CNN_no_ascon_model_metrics_gateway_*.csv",
    ),
]


EXPECTED_ROUNDS_PER_ATTEMPT = 30
# Las figuras del documento se cortan en esta ronda para mantener homogeneidad
# visual entre variantes (algunas corridas RN llegaron hasta 30 rondas, mientras
# que el resto se quedo en 20). Las tablas SRE conservan todas las rondas.
MAX_ROUND_FOR_PLOTS = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _parse_time_only(series: pd.Series, anchor_date: pd.Timestamp) -> pd.Series:
    anchor_str = anchor_date.strftime("%Y-%m-%d")
    return pd.to_datetime(anchor_str + " " + series.astype(str), errors="coerce")


def _normalize_transport_columns(df: pd.DataFrame, security: str) -> pd.DataFrame:
    """Asegura que `pt_bytes`, `enc_bytes`, `overhead_bytes` existan.

    Para variantes plain (sin cifrado), `payload_bytes` representa el tamano
    enviado/recibido. Se replica como pt_bytes y enc_bytes con overhead 0.
    """
    if df.empty:
        return df
    if security == "PLAIN":
        if "payload_bytes" in df.columns:
            df["pt_bytes"] = df["payload_bytes"]
            df["enc_bytes"] = df["payload_bytes"]
            df["overhead_bytes"] = 0
    for col in ["pt_bytes", "enc_bytes", "overhead_bytes"]:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def _normalize_operation(df: pd.DataFrame) -> pd.DataFrame:
    """Mapea operaciones plain (deserialize/serialize) a decrypt/encrypt
    para mantener la semantica unificada en los SLI.
    """
    if df.empty or "operation" not in df.columns:
        return df
    df = df.copy()
    df["operation_raw"] = df["operation"]
    df["operation"] = df["operation"].replace({
        "deserialize": "decrypt",
        "serialize": "encrypt",
    })
    return df


def _normalize_channel(df: pd.DataFrame) -> pd.DataFrame:
    """Unifica los nombres de canal entre variantes estandar y FOG.

    En FOG el servidor reporta `RPi_leader->PC` y `PC->RPi_leader`. Para
    mantener la semantica unificada se mapean a `RPi->PC` y `PC->RPi`,
    preservando el original en `channel_raw`.
    """
    if df.empty or "channel" not in df.columns:
        return df
    df = df.copy()
    df["channel_raw"] = df["channel"]
    df["channel"] = df["channel"].replace({
        "RPi_leader->PC": "RPi->PC",
        "PC->RPi_leader": "PC->RPi",
        "RPi_peer->RPi_leader": "RPi_peer->RPi_leader",   # se conserva tal cual
        "RPi_leader->RPi_peer": "RPi_leader->RPi_peer",
    })
    return df


def _fix_time_rollover(df: pd.DataFrame, group_cols: list[str], time_col: str) -> pd.DataFrame:
    """Si los timestamps cruzan medianoche (van hacia atras), agrega 1 dia
    a las filas posteriores al rollover. Trabaja por grupo independiente.
    """
    if df.empty or time_col not in df.columns:
        return df
    df = df.copy()
    parts: list[pd.DataFrame] = []
    for _, group in df.groupby(group_cols, sort=False, dropna=False):
        sorted_group = group.sort_index()
        times = sorted_group[time_col].tolist()
        offset_days = 0
        new_times = []
        prev = None
        for t in times:
            if pd.isna(t):
                new_times.append(t)
                continue
            current = t + pd.Timedelta(days=offset_days)
            if prev is not None and current < prev:
                offset_days += 1
                current = t + pd.Timedelta(days=offset_days)
            new_times.append(current)
            prev = current
        sorted_group[time_col] = new_times
        parts.append(sorted_group)
    fixed = pd.concat(parts).sort_index()
    return fixed


# ---------------------------------------------------------------------------
# Carga por variante
# ---------------------------------------------------------------------------

def _discover_attempt_dirs(results_dir: Path) -> list[tuple[int, Path]]:
    if not results_dir.exists():
        return []
    discovered: list[tuple[int, Path]] = []
    for path in results_dir.iterdir():
        if not path.is_dir():
            continue
        match = ATTEMPT_RE.match(path.name)
        if match:
            discovered.append((int(match.group(1)), path))
    return sorted(discovered, key=lambda item: item[0])


def _load_attempt(spec: VariantSpec, attempt_id: int, attempt_dir: Path) -> dict[str, pd.DataFrame]:
    gw_paths = sorted(attempt_dir.glob(spec.gateway_transport_glob))
    srv_paths = sorted(attempt_dir.glob(spec.server_transport_glob))
    mm_paths = sorted(attempt_dir.glob(spec.model_metrics_glob))
    gh_paths = sorted(attempt_dir.glob(spec.global_history_glob))

    def _stamp(df: pd.DataFrame, scope: str, source_file: str) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df["variant"] = spec.label
        df["model"] = spec.model
        df["security"] = spec.security
        df["topology"] = spec.topology
        df["attempt_id"] = attempt_id
        df["attempt_dir"] = str(attempt_dir)
        df["source_file"] = source_file
        df["source_scope"] = scope
        return df

    gateway_transport = pd.concat(
        [_stamp(_read_csv(p), "gateway_transport", p.name) for p in gw_paths] or [pd.DataFrame()],
        ignore_index=True, sort=False,
    )
    server_transport = pd.concat(
        [_stamp(_read_csv(p), "server_transport", p.name) for p in srv_paths] or [pd.DataFrame()],
        ignore_index=True, sort=False,
    )
    model_metrics = pd.concat(
        [_stamp(_read_csv(p), "local_training", p.name) for p in mm_paths] or [pd.DataFrame()],
        ignore_index=True, sort=False,
    )
    global_history = pd.concat(
        [_stamp(_read_csv(p), "global_round", p.name) for p in gh_paths] or [pd.DataFrame()],
        ignore_index=True, sort=False,
    )

    gateway_transport = _normalize_transport_columns(gateway_transport, spec.security)
    server_transport = _normalize_transport_columns(server_transport, spec.security)
    gateway_transport = _normalize_operation(gateway_transport)
    server_transport = _normalize_operation(server_transport)
    gateway_transport = _normalize_channel(gateway_transport)
    server_transport = _normalize_channel(server_transport)

    anchor_date = pd.Timestamp("2026-01-01")
    if not model_metrics.empty:
        parsed = pd.to_datetime(model_metrics["timestamp"], errors="coerce")
        if parsed.notna().any():
            anchor_date = parsed.dropna().min().normalize()
            model_metrics["timestamp_dt"] = parsed
    if "timestamp_dt" not in model_metrics.columns and not model_metrics.empty:
        model_metrics["timestamp_dt"] = pd.to_datetime(
            model_metrics.get("timestamp"), errors="coerce"
        )

    def _col(df: pd.DataFrame, name: str, default: Any = pd.NA) -> pd.Series:
        if name in df.columns:
            return df[name]
        return pd.Series([default] * len(df), index=df.index)

    if not gateway_transport.empty:
        gateway_transport["timestamp_dt"] = _parse_time_only(
            _col(gateway_transport, "timestamp"), anchor_date
        )
        gateway_transport["event_type"] = "transport.crypto"
        gateway_transport["round_ref"] = (
            pd.to_numeric(_col(gateway_transport, "fl_round"), errors="coerce") + 1
        )
        gateway_transport["gateway_id"] = _col(gateway_transport, "device_suffix").astype(str)
        gateway_transport["node_id"] = _col(gateway_transport, "device_suffix").astype(str)
        if "device_suffix" not in gateway_transport.columns:
            gateway_transport["device_suffix"] = "unknown"
        if "channel" not in gateway_transport.columns:
            gateway_transport["channel"] = "unknown"
        if "operation" not in gateway_transport.columns:
            gateway_transport["operation"] = "unknown"
        if "elapsed_ms" not in gateway_transport.columns:
            gateway_transport["elapsed_ms"] = pd.NA

    if not server_transport.empty:
        server_transport["timestamp_dt"] = _parse_time_only(
            _col(server_transport, "timestamp"), anchor_date
        )
        server_transport["event_type"] = "transport.crypto"
        flr = pd.to_numeric(_col(server_transport, "fl_round"), errors="coerce")
        if "channel" not in server_transport.columns:
            server_transport["channel"] = "unknown"
        if "operation" not in server_transport.columns:
            server_transport["operation"] = "unknown"
        if "elapsed_ms" not in server_transport.columns:
            server_transport["elapsed_ms"] = pd.NA
        server_transport["round_ref"] = np.where(
            (server_transport["channel"] == "PC->RPi") & (server_transport["operation"] == "encrypt"),
            flr,
            flr + 1,
        )
        server_transport["gateway_id"] = "aggregate"
        if "device_suffix" not in server_transport.columns:
            server_transport["device_suffix"] = "server"
        server_transport["node_id"] = server_transport["device_suffix"].astype(str)

    if not model_metrics.empty:
        model_metrics["event_type"] = "model.local_train"
        model_metrics["round_ref"] = (
            pd.to_numeric(_col(model_metrics, "fl_round"), errors="coerce") + 1
        )
        if "device_suffix" not in model_metrics.columns:
            model_metrics["device_suffix"] = "unknown"
        if "stage" not in model_metrics.columns:
            model_metrics["stage"] = "local_train"
        model_metrics["gateway_id"] = model_metrics["device_suffix"].astype(str)
        model_metrics["node_id"] = model_metrics["device_suffix"].astype(str)

    if not global_history.empty:
        global_history["timestamp_dt"] = _parse_time_only(
            global_history["time"], anchor_date
        )
        global_history["event_type"] = "model.global_round"
        global_history["round_ref"] = pd.to_numeric(
            global_history["round"], errors="coerce"
        )
        global_history["gateway_id"] = "aggregate"
        global_history["node_id"] = "server"

    return {
        "gateway_transport": gateway_transport,
        "server_transport": server_transport,
        "model_metrics": model_metrics,
        "global_history": global_history,
    }


def load_all() -> dict[str, pd.DataFrame]:
    gateway_parts: list[pd.DataFrame] = []
    server_parts: list[pd.DataFrame] = []
    model_parts: list[pd.DataFrame] = []
    global_parts: list[pd.DataFrame] = []
    inventory_rows: list[dict[str, Any]] = []

    for spec in VARIANTS:
        attempts = _discover_attempt_dirs(spec.results_dir)
        if not attempts:
            print(f"[WARN] variant={spec.label} sin intentos en {spec.results_dir}")
            continue
        for attempt_id, attempt_dir in attempts:
            frames = _load_attempt(spec, attempt_id, attempt_dir)
            gateway_parts.append(frames["gateway_transport"])
            server_parts.append(frames["server_transport"])
            model_parts.append(frames["model_metrics"])
            global_parts.append(frames["global_history"])
            inventory_rows.append({
                "variant": spec.label,
                "model": spec.model,
                "security": spec.security,
                "topology": spec.topology,
                "attempt_id": attempt_id,
                "attempt_dir": str(attempt_dir),
                "gateway_transport_files": int(len(list(attempt_dir.glob(spec.gateway_transport_glob)))),
                "server_transport_files": int(len(list(attempt_dir.glob(spec.server_transport_glob)))),
                "model_metric_files": int(len(list(attempt_dir.glob(spec.model_metrics_glob)))),
                "global_history_files": int(len(list(attempt_dir.glob(spec.global_history_glob)))),
            })

    gateway_transport = pd.concat(gateway_parts, ignore_index=True, sort=False) if gateway_parts else pd.DataFrame()
    server_transport = pd.concat(server_parts, ignore_index=True, sort=False) if server_parts else pd.DataFrame()
    model_metrics = pd.concat(model_parts, ignore_index=True, sort=False) if model_parts else pd.DataFrame()
    global_history = pd.concat(global_parts, ignore_index=True, sort=False) if global_parts else pd.DataFrame()
    attempt_inventory = pd.DataFrame(inventory_rows).sort_values(
        ["variant", "attempt_id"]
    ).reset_index(drop=True)

    return {
        "attempt_inventory": attempt_inventory,
        "gateway_transport": gateway_transport,
        "server_transport": server_transport,
        "model_metrics": model_metrics,
        "global_history": global_history,
    }


# ---------------------------------------------------------------------------
# Eventos canonicos
# ---------------------------------------------------------------------------

def build_canonical_log_events(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for key, event_name in [
        ("gateway_transport", "transport.crypto.gateway"),
        ("server_transport", "transport.crypto.server"),
        ("model_metrics", "model.local_train"),
        ("global_history", "model.global_round"),
    ]:
        df = data[key].copy()
        if df.empty:
            continue
        df["event_name"] = event_name
        df["status"] = "reconstructed_from_csv"
        df["trace_id"] = (
            "variant:" + df["variant"].astype(str)
            + ":attempt:" + df["attempt_id"].astype(str)
            + ":round:" + df["round_ref"].astype("Int64").astype(str)
        )
        parts.append(df)
    canonical = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
    if not canonical.empty:
        canonical = canonical.sort_values(
            ["variant", "attempt_id", "timestamp_dt", "event_name"]
        ).reset_index(drop=True)
    return canonical


# ---------------------------------------------------------------------------
# SLI summaries
# ---------------------------------------------------------------------------

def compute_transport_sli(transport_events: pd.DataFrame) -> pd.DataFrame:
    if transport_events.empty:
        return pd.DataFrame()
    df = transport_events.copy()
    df["expansion_ratio"] = (
        pd.to_numeric(df["enc_bytes"], errors="coerce") /
        pd.to_numeric(df["pt_bytes"], errors="coerce")
    )
    df["overhead_ratio_pct"] = (
        pd.to_numeric(df["overhead_bytes"], errors="coerce") /
        pd.to_numeric(df["pt_bytes"], errors="coerce") * 100.0
    )
    summary = (
        df.groupby(
            ["variant", "model", "security", "topology",
             "attempt_id", "source_scope", "device_suffix",
             "channel", "operation"],
            dropna=False,
        )
        .apply(lambda g: pd.Series({
            "events": int(len(g)),
            "avg_ms": _safe_mean(g["elapsed_ms"]),
            "median_ms": _safe_pct(g["elapsed_ms"], 50),
            "p95_ms": _safe_pct(g["elapsed_ms"], 95),
            "max_ms": pd.to_numeric(g["elapsed_ms"], errors="coerce").max(),
            "avg_pt_bytes": _safe_mean(g["pt_bytes"]),
            "avg_enc_bytes": _safe_mean(g["enc_bytes"]),
            "avg_overhead_bytes": _safe_mean(g["overhead_bytes"]),
            "avg_overhead_ratio_pct": _safe_mean(g["overhead_ratio_pct"]),
            "avg_expansion_ratio": _safe_mean(g["expansion_ratio"]),
        }))
        .reset_index()
        .sort_values(["variant", "attempt_id", "source_scope", "channel", "operation"])
        .reset_index(drop=True)
    )
    return summary


def compute_local_training_sli(model_metrics: pd.DataFrame) -> pd.DataFrame:
    if model_metrics.empty:
        return pd.DataFrame()
    df = model_metrics.loc[model_metrics["stage"] == "local_train"].copy()
    if df.empty:
        return pd.DataFrame()
    summary = (
        df.groupby(
            ["variant", "model", "security", "topology",
             "attempt_id", "device_suffix"],
            dropna=False,
        )
        .apply(lambda g: pd.Series({
            "rounds_seen": int(g["fl_round"].nunique()),
            "avg_accuracy": _safe_mean(g["accuracy"]),
            "std_accuracy": _safe_std(g["accuracy"]),
            "min_accuracy": pd.to_numeric(g["accuracy"], errors="coerce").min(),
            "max_accuracy": pd.to_numeric(g["accuracy"], errors="coerce").max(),
            "avg_loss": _safe_mean(g["loss"]),
            "std_loss": _safe_std(g["loss"]),
            "min_loss": pd.to_numeric(g["loss"], errors="coerce").min(),
            "max_loss": pd.to_numeric(g["loss"], errors="coerce").max(),
            "avg_num_samples": _safe_mean(g["num_samples"]),
            "avg_buffer_target": _safe_mean(g.get("buffer_target", pd.Series([np.nan]))),
        }))
        .reset_index()
        .sort_values(["variant", "attempt_id", "device_suffix"])
        .reset_index(drop=True)
    )
    return summary


def compute_global_round_sli(
    global_history: pd.DataFrame,
    expected_rounds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if global_history.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = global_history.copy().sort_values(["variant", "attempt_id", "round"]).reset_index(drop=True)
    # corrige rollovers de medianoche dentro de cada (variant, attempt_id)
    df = _fix_time_rollover(df, ["variant", "attempt_id"], "timestamp_dt").reset_index(drop=True)
    df["round_duration_sec"] = (
        df.groupby(["variant", "attempt_id"])["timestamp_dt"].diff().dt.total_seconds()
    )
    for col in ["w3_mag", "w4_normal", "w4_brute", "w4_scan"]:
        if col in df.columns:
            df[f"{col}_drift_abs"] = (
                df.groupby(["variant", "attempt_id"])[col].diff().abs()
            )
        else:
            df[f"{col}_drift_abs"] = pd.NA
    drift_cols = [f"{c}_drift_abs" for c in ["w3_mag", "w4_normal", "w4_brute", "w4_scan"]]
    df["weight_drift_total_abs"] = (
        df[drift_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
    )
    summary = (
        df.groupby(
            ["variant", "model", "security", "topology", "attempt_id"],
            dropna=False,
        )
        .apply(lambda g: pd.Series({
            "rounds_observed": int(g["round"].nunique()),
            "completion_rate": float(g["round"].nunique() / expected_rounds),
            "last_accuracy": pd.to_numeric(g["accuracy"], errors="coerce").iloc[-1],
            "last_loss": pd.to_numeric(g["loss"], errors="coerce").iloc[-1],
            "best_accuracy": pd.to_numeric(g["accuracy"], errors="coerce").max(),
            "min_loss": pd.to_numeric(g["loss"], errors="coerce").min(),
            "avg_round_duration_sec": _safe_mean(g["round_duration_sec"]),
            "p95_round_duration_sec": _safe_pct(g["round_duration_sec"], 95),
            "avg_weight_drift_total_abs": _safe_mean(g["weight_drift_total_abs"]),
        }))
        .reset_index()
        .sort_values(["variant", "attempt_id"])
        .reset_index(drop=True)
    )
    return summary, df


def compute_gateway_alignment(
    model_metrics: pd.DataFrame,
    global_history: pd.DataFrame,
) -> pd.DataFrame:
    if model_metrics.empty:
        return pd.DataFrame()
    local_df = model_metrics.loc[model_metrics["stage"] == "local_train"].copy()
    if local_df.empty:
        return pd.DataFrame()
    local_df["global_round"] = (
        pd.to_numeric(local_df["fl_round"], errors="coerce") + 1
    )

    rows: list[dict[str, Any]] = []
    for (variant, attempt_id), group in local_df.groupby(["variant", "attempt_id"]):
        gateways = sorted(group["device_suffix"].dropna().unique().tolist())
        if len(gateways) < 2:
            continue
        gw_a, gw_b = gateways[0], gateways[1]
        for global_round in sorted(group["global_round"].dropna().unique().tolist()):
            sub = group.loc[group["global_round"] == global_round]
            row_a = sub.loc[sub["device_suffix"] == gw_a]
            row_b = sub.loc[sub["device_suffix"] == gw_b]
            if row_a.empty or row_b.empty:
                continue
            acc_a = pd.to_numeric(row_a["accuracy"], errors="coerce").iloc[0]
            acc_b = pd.to_numeric(row_b["accuracy"], errors="coerce").iloc[0]
            loss_a = pd.to_numeric(row_a["loss"], errors="coerce").iloc[0]
            loss_b = pd.to_numeric(row_b["loss"], errors="coerce").iloc[0]
            ts_a = pd.to_datetime(row_a["timestamp_dt"], errors="coerce").iloc[0]
            ts_b = pd.to_datetime(row_b["timestamp_dt"], errors="coerce").iloc[0]
            rows.append({
                "variant": variant,
                "attempt_id": attempt_id,
                "global_round": int(global_round),
                "gateway_a": gw_a,
                "gateway_b": gw_b,
                f"accuracy_{gw_a}": acc_a,
                f"accuracy_{gw_b}": acc_b,
                "accuracy_skew_abs": abs(acc_a - acc_b) if pd.notna(acc_a) and pd.notna(acc_b) else np.nan,
                f"loss_{gw_a}": loss_a,
                f"loss_{gw_b}": loss_b,
                "loss_skew_abs": abs(loss_a - loss_b) if pd.notna(loss_a) and pd.notna(loss_b) else np.nan,
                "gateway_train_skew_sec": (
                    abs((ts_a - ts_b).total_seconds())
                    if pd.notna(ts_a) and pd.notna(ts_b) else np.nan
                ),
            })
    alignment = pd.DataFrame(rows)
    if alignment.empty:
        return alignment
    if not global_history.empty:
        join = global_history[["variant", "attempt_id", "round", "accuracy", "loss"]].rename(
            columns={"round": "global_round", "accuracy": "global_accuracy", "loss": "global_loss"}
        )
        alignment = alignment.merge(join, on=["variant", "attempt_id", "global_round"], how="left")
    return alignment.sort_values(["variant", "attempt_id", "global_round"]).reset_index(drop=True)


def compute_class_mix(gateway_transport: pd.DataFrame) -> pd.DataFrame:
    if gateway_transport.empty or "sample_label_name" not in gateway_transport.columns:
        return pd.DataFrame()
    df = gateway_transport.copy()
    counts = (
        df.groupby(["variant", "attempt_id", "device_suffix", "sample_label_name"], dropna=False)
        .size()
        .reset_index(name="samples")
    )
    totals = counts.groupby(["variant", "attempt_id", "device_suffix"])["samples"].transform("sum")
    counts["share_pct"] = counts["samples"] / totals * 100.0
    return counts.sort_values(
        ["variant", "attempt_id", "device_suffix", "sample_label_name"]
    ).reset_index(drop=True)


def build_round_trace_summary(
    gateway_transport: pd.DataFrame,
    server_transport: pd.DataFrame,
    model_metrics: pd.DataFrame,
    global_history: pd.DataFrame,
) -> pd.DataFrame:
    if global_history.empty:
        return pd.DataFrame()
    local_df = model_metrics.loc[model_metrics["stage"] == "local_train"].copy() if not model_metrics.empty else pd.DataFrame()
    if not local_df.empty:
        local_df["global_round"] = (
            pd.to_numeric(local_df["fl_round"], errors="coerce") + 1
        )

    if not gateway_transport.empty:
        gateway_batch = (
            gateway_transport.groupby(
                ["variant", "attempt_id", "device_suffix", "round_ref"],
                dropna=False,
            )
            .apply(lambda g: pd.Series({
                "gateway_events": int(len(g)),
                "gateway_decrypt_avg_ms": _safe_mean(g["elapsed_ms"]),
                "gateway_decrypt_p95_ms": _safe_pct(g["elapsed_ms"], 95),
                "gateway_avg_overhead_bytes": _safe_mean(g["overhead_bytes"]),
                "gateway_samples_seen": int(g["client_id"].notna().sum()) if "client_id" in g.columns else int(len(g)),
            }))
            .reset_index()
            .rename(columns={"round_ref": "global_round"})
        )
    else:
        gateway_batch = pd.DataFrame()

    if not local_df.empty:
        local_rounds = (
            local_df.groupby(
                ["variant", "attempt_id", "device_suffix", "global_round"],
                dropna=False,
            )
            .agg(
                local_accuracy=("accuracy", "first"),
                local_loss=("loss", "first"),
                local_num_samples=("num_samples", "first"),
                local_timestamp=("timestamp_dt", "first"),
            )
            .reset_index()
        )
    else:
        local_rounds = pd.DataFrame()

    if not server_transport.empty:
        server_rounds = (
            server_transport.groupby(
                ["variant", "attempt_id", "round_ref", "channel", "operation"],
                dropna=False,
            )
            .apply(lambda g: pd.Series({
                "server_events": int(len(g)),
                "server_avg_ms": _safe_mean(g["elapsed_ms"]),
                "server_p95_ms": _safe_pct(g["elapsed_ms"], 95),
                "server_avg_overhead_bytes": _safe_mean(g["overhead_bytes"]),
            }))
            .reset_index()
            .rename(columns={"round_ref": "global_round"})
        )
        decrypt_server = server_rounds.loc[
            (server_rounds["channel"] == "RPi->PC") & (server_rounds["operation"] == "decrypt")
        ].rename(columns={
            "server_events": "server_rpi_pc_events",
            "server_avg_ms": "server_rpi_pc_avg_ms",
            "server_p95_ms": "server_rpi_pc_p95_ms",
            "server_avg_overhead_bytes": "server_rpi_pc_avg_overhead_bytes",
        })
        encrypt_server = server_rounds.loc[
            (server_rounds["channel"] == "PC->RPi") & (server_rounds["operation"] == "encrypt")
        ].rename(columns={
            "server_events": "server_pc_rpi_events",
            "server_avg_ms": "server_pc_rpi_avg_ms",
            "server_p95_ms": "server_pc_rpi_p95_ms",
            "server_avg_overhead_bytes": "server_pc_rpi_avg_overhead_bytes",
        })
    else:
        decrypt_server = pd.DataFrame()
        encrypt_server = pd.DataFrame()

    traces: list[dict[str, Any]] = []
    sorted_global = global_history.sort_values(["variant", "attempt_id", "round"])
    for _, row in sorted_global.iterrows():
        variant = row["variant"]
        attempt_id = int(row["attempt_id"])
        global_round = int(row["round"])
        trace_id = f"variant:{variant}:attempt:{attempt_id}:round:{global_round}"
        base: dict[str, Any] = {
            "trace_id": trace_id,
            "variant": variant,
            "model": row.get("model"),
            "security": row.get("security"),
            "topology": row.get("topology"),
            "attempt_id": attempt_id,
            "global_round": global_round,
            "global_timestamp": row.get("timestamp_dt"),
            "global_accuracy": row.get("accuracy"),
            "global_loss": row.get("loss"),
            "w3_mag": row.get("w3_mag"),
            "w4_normal": row.get("w4_normal"),
            "w4_brute": row.get("w4_brute"),
            "w4_scan": row.get("w4_scan"),
        }
        gateways = []
        if not local_df.empty:
            gateways = sorted(
                local_df.loc[
                    (local_df["variant"] == variant) & (local_df["attempt_id"] == attempt_id),
                    "device_suffix",
                ].dropna().unique().tolist()
            )
        for gw in gateways:
            if not local_rounds.empty:
                lr = local_rounds.loc[
                    (local_rounds["variant"] == variant) &
                    (local_rounds["attempt_id"] == attempt_id) &
                    (local_rounds["device_suffix"] == gw) &
                    (local_rounds["global_round"] == global_round)
                ]
                if not lr.empty:
                    base[f"{gw}_local_accuracy"] = lr["local_accuracy"].iloc[0]
                    base[f"{gw}_local_loss"] = lr["local_loss"].iloc[0]
                    base[f"{gw}_local_num_samples"] = lr["local_num_samples"].iloc[0]
                    base[f"{gw}_local_timestamp"] = lr["local_timestamp"].iloc[0]
            if not gateway_batch.empty:
                gb = gateway_batch.loc[
                    (gateway_batch["variant"] == variant) &
                    (gateway_batch["attempt_id"] == attempt_id) &
                    (gateway_batch["device_suffix"] == gw) &
                    (gateway_batch["global_round"] == global_round)
                ]
                if not gb.empty:
                    base[f"{gw}_gateway_events"] = gb["gateway_events"].iloc[0]
                    base[f"{gw}_gateway_decrypt_avg_ms"] = gb["gateway_decrypt_avg_ms"].iloc[0]
                    base[f"{gw}_gateway_decrypt_p95_ms"] = gb["gateway_decrypt_p95_ms"].iloc[0]
                    base[f"{gw}_gateway_avg_overhead_bytes"] = gb["gateway_avg_overhead_bytes"].iloc[0]
                    base[f"{gw}_gateway_samples_seen"] = gb["gateway_samples_seen"].iloc[0]
        if not decrypt_server.empty:
            sd = decrypt_server.loc[
                (decrypt_server["variant"] == variant) &
                (decrypt_server["attempt_id"] == attempt_id) &
                (decrypt_server["global_round"] == global_round)
            ]
            if not sd.empty:
                for col in ["server_rpi_pc_events", "server_rpi_pc_avg_ms",
                            "server_rpi_pc_p95_ms", "server_rpi_pc_avg_overhead_bytes"]:
                    base[col] = sd[col].iloc[0]
        if not encrypt_server.empty:
            se = encrypt_server.loc[
                (encrypt_server["variant"] == variant) &
                (encrypt_server["attempt_id"] == attempt_id) &
                (encrypt_server["global_round"] == global_round)
            ]
            if not se.empty:
                for col in ["server_pc_rpi_events", "server_pc_rpi_avg_ms",
                            "server_pc_rpi_p95_ms", "server_pc_rpi_avg_overhead_bytes"]:
                    base[col] = se[col].iloc[0]
        traces.append(base)
    return pd.DataFrame(traces).sort_values(["variant", "attempt_id", "global_round"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Executive summary y plots
# ---------------------------------------------------------------------------

def compute_executive_summary(
    attempt_inventory: pd.DataFrame,
    transport_summary: pd.DataFrame,
    local_summary: pd.DataFrame,
    global_summary: pd.DataFrame,
    class_mix: pd.DataFrame,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "attempts_analyzed_total": int(len(attempt_inventory)),
        "variants": {},
        "global": {},
    }
    if attempt_inventory.empty:
        return summary

    for variant in sorted(attempt_inventory["variant"].unique().tolist()):
        inv_v = attempt_inventory.loc[attempt_inventory["variant"] == variant]
        ts_v = transport_summary.loc[transport_summary["variant"] == variant] if not transport_summary.empty else pd.DataFrame()
        ls_v = local_summary.loc[local_summary["variant"] == variant] if not local_summary.empty else pd.DataFrame()
        gs_v = global_summary.loc[global_summary["variant"] == variant] if not global_summary.empty else pd.DataFrame()
        gw_p95 = ts_v.loc[
            (ts_v["source_scope"] == "gateway_transport") &
            (ts_v["channel"] == "ESP32->RPi") &
            (ts_v["operation"] == "decrypt"), "p95_ms"
        ] if not ts_v.empty else pd.Series(dtype=float)
        srv_dec_p95 = ts_v.loc[
            (ts_v["source_scope"] == "server_transport") &
            (ts_v["channel"] == "RPi->PC") &
            (ts_v["operation"] == "decrypt"), "p95_ms"
        ] if not ts_v.empty else pd.Series(dtype=float)
        srv_enc_p95 = ts_v.loc[
            (ts_v["source_scope"] == "server_transport") &
            (ts_v["channel"] == "PC->RPi") &
            (ts_v["operation"] == "encrypt"), "p95_ms"
        ] if not ts_v.empty else pd.Series(dtype=float)
        summary["variants"][variant] = {
            "attempts_analyzed": int(len(inv_v)),
            "attempt_range": [
                int(inv_v["attempt_id"].min()),
                int(inv_v["attempt_id"].max()),
            ],
            "avg_gateway_decrypt_p95_ms": _safe_mean(gw_p95),
            "avg_server_decrypt_p95_ms": _safe_mean(srv_dec_p95),
            "avg_server_encrypt_p95_ms": _safe_mean(srv_enc_p95),
            "avg_payload_overhead_bytes": _safe_mean(ts_v["avg_overhead_bytes"]) if not ts_v.empty else float("nan"),
            "avg_payload_overhead_ratio_pct": _safe_mean(ts_v["avg_overhead_ratio_pct"]) if not ts_v.empty else float("nan"),
            "avg_local_accuracy": _safe_mean(ls_v["avg_accuracy"]) if not ls_v.empty else float("nan"),
            "avg_local_loss": _safe_mean(ls_v["avg_loss"]) if not ls_v.empty else float("nan"),
            "avg_global_last_accuracy": _safe_mean(gs_v["last_accuracy"]) if not gs_v.empty else float("nan"),
            "avg_global_last_loss": _safe_mean(gs_v["last_loss"]) if not gs_v.empty else float("nan"),
            "avg_round_completion_rate": _safe_mean(gs_v["completion_rate"]) if not gs_v.empty else float("nan"),
            "avg_round_duration_sec": _safe_mean(gs_v["avg_round_duration_sec"]) if not gs_v.empty else float("nan"),
        }

    summary["global"] = {
        "variants_count": int(attempt_inventory["variant"].nunique()),
        "attempt_range_total": [
            int(attempt_inventory["attempt_id"].min()),
            int(attempt_inventory["attempt_id"].max()),
        ],
        "avg_global_last_accuracy": _safe_mean(global_summary["last_accuracy"]) if not global_summary.empty else float("nan"),
        "avg_global_last_loss": _safe_mean(global_summary["last_loss"]) if not global_summary.empty else float("nan"),
        "avg_payload_overhead_bytes": _safe_mean(transport_summary["avg_overhead_bytes"]) if not transport_summary.empty else float("nan"),
        "avg_round_duration_sec": _safe_mean(global_summary["avg_round_duration_sec"]) if not global_summary.empty else float("nan"),
        "class_labels_observed": sorted(
            class_mix["sample_label_name"].dropna().astype(str).unique().tolist()
        ) if not class_mix.empty else [],
    }
    return summary


def _save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def _save_plot(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def generate_plots(
    output_dir: Path,
    transport_summary: pd.DataFrame,
    alignment: pd.DataFrame,
    global_history_enriched: pd.DataFrame,
    class_mix: pd.DataFrame,
) -> dict[str, str]:
    plot_paths: dict[str, str] = {}

    # 1. transport_latency_p95 -- p95 promedio por variante y canal
    if not transport_summary.empty:
        chart = (
            transport_summary
            .groupby(["variant", "channel", "operation"])["p95_ms"]
            .mean()
            .reset_index()
        )
        chart["channel_op"] = chart["channel"] + " | " + chart["operation"]
        pivot = chart.pivot(index="variant", columns="channel_op", values="p95_ms")
        fig, ax = plt.subplots(figsize=(13, 6))
        pivot.plot(kind="bar", ax=ax)
        ax.set_title("Latencia p95 (ms) promedio por variante y canal")
        ax.set_xlabel("Variante")
        ax.set_ylabel("p95 (ms)")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title="Canal | Operacion", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        path = output_dir / "transport_latency_p95.png"
        _save_plot(fig, path)
        plot_paths["transport_latency_p95"] = str(path)

        # 2. transport_payload_bytes
        bytes_chart = (
            transport_summary
            .groupby(["variant", "channel"])[["avg_pt_bytes", "avg_enc_bytes", "avg_overhead_bytes"]]
            .mean()
            .reset_index()
        )
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        # Plain bytes
        pivot_pt = bytes_chart.pivot(index="variant", columns="channel", values="avg_pt_bytes")
        pivot_pt.plot(kind="bar", ax=axes[0])
        axes[0].set_title("Tamano promedio de plaintext (bytes)")
        axes[0].set_xlabel("Variante")
        axes[0].set_ylabel("Bytes")
        axes[0].tick_params(axis="x", rotation=30)
        axes[0].legend(title="Canal", fontsize=8)
        # Overhead
        pivot_oh = bytes_chart.pivot(index="variant", columns="channel", values="avg_overhead_bytes")
        pivot_oh.plot(kind="bar", ax=axes[1], color=["#ff7f0e", "#2ca02c", "#d62728", "#9467bd"][:len(pivot_oh.columns)])
        axes[1].set_title("Overhead promedio (bytes) sobre plaintext")
        axes[1].set_xlabel("Variante")
        axes[1].set_ylabel("Bytes")
        axes[1].tick_params(axis="x", rotation=30)
        axes[1].legend(title="Canal", fontsize=8)
        path = output_dir / "transport_payload_bytes.png"
        _save_plot(fig, path)
        plot_paths["transport_payload_bytes"] = str(path)

    # 3. global_accuracy_loss -- una linea por variante (corte a MAX_ROUND_FOR_PLOTS)
    if not global_history_enriched.empty:
        gh_plot = global_history_enriched.loc[
            global_history_enriched["round"] <= MAX_ROUND_FOR_PLOTS
        ].copy()
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True)
        for variant, group in gh_plot.groupby("variant"):
            by_round = group.groupby("round")[["accuracy", "loss"]].mean().reset_index()
            axes[0].plot(by_round["round"], by_round["accuracy"], marker="o", label=variant, alpha=0.85)
            axes[1].plot(by_round["round"], by_round["loss"], marker="s", label=variant, alpha=0.85)
        axes[0].set_title(f"Accuracy global promedio por ronda (1-{MAX_ROUND_FOR_PLOTS})")
        axes[0].set_xlabel("Ronda global")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_xlim(1, MAX_ROUND_FOR_PLOTS)
        axes[0].legend(fontsize=8, loc="lower right")
        axes[1].set_title(f"Loss global promedio por ronda (1-{MAX_ROUND_FOR_PLOTS})")
        axes[1].set_xlabel("Ronda global")
        axes[1].set_ylabel("Loss")
        axes[1].set_xlim(1, MAX_ROUND_FOR_PLOTS)
        axes[1].legend(fontsize=8, loc="upper right")
        path = output_dir / "global_accuracy_loss.png"
        _save_plot(fig, path)
        plot_paths["global_accuracy_loss"] = str(path)

        # 4. weight_magnitude_trends
        weight_cols = [c for c in ["w3_mag", "w4_normal", "w4_brute", "w4_scan"]
                       if c in gh_plot.columns]
        if weight_cols:
            n = len(weight_cols)
            fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharex=True)
            if n == 1:
                axes = [axes]
            for ax, col in zip(axes, weight_cols):
                for variant, group in gh_plot.groupby("variant"):
                    by_round = group.groupby("round")[col].mean().reset_index()
                    ax.plot(by_round["round"], by_round[col], marker="o", label=variant, alpha=0.85)
                ax.set_title(col)
                ax.set_xlabel("Ronda global")
                ax.set_ylabel("Magnitud")
                ax.set_xlim(1, MAX_ROUND_FOR_PLOTS)
                ax.legend(fontsize=7, loc="best")
            path = output_dir / "weight_magnitude_trends.png"
            _save_plot(fig, path)
            plot_paths["weight_magnitude_trends"] = str(path)

        # 5. round_duration
        fig, ax = plt.subplots(figsize=(12, 6))
        for variant, group in gh_plot.groupby("variant"):
            duration = group.groupby("round")["round_duration_sec"].mean().reset_index().dropna()
            if duration.empty:
                continue
            ax.plot(duration["round"], duration["round_duration_sec"],
                    marker="o", label=variant, alpha=0.85)
        ax.set_title(f"Duracion promedio por ronda (s) -- rondas 1-{MAX_ROUND_FOR_PLOTS}")
        ax.set_xlabel("Ronda global")
        ax.set_ylabel("Duracion (s)")
        ax.set_xlim(1, MAX_ROUND_FOR_PLOTS)
        ax.legend(fontsize=8, loc="best")
        path = output_dir / "round_duration.png"
        _save_plot(fig, path)
        plot_paths["round_duration"] = str(path)

    # 6. gateway_accuracy_skew
    if not alignment.empty:
        align_plot = alignment.loc[alignment["global_round"] <= MAX_ROUND_FOR_PLOTS].copy()
        fig, ax = plt.subplots(figsize=(12, 6))
        for variant, group in align_plot.groupby("variant"):
            skew = group.groupby("global_round")["accuracy_skew_abs"].mean().reset_index()
            ax.plot(skew["global_round"], skew["accuracy_skew_abs"],
                    marker="o", label=variant, alpha=0.85)
        ax.set_title(f"Skew promedio de accuracy entre gateways por variante (1-{MAX_ROUND_FOR_PLOTS})")
        ax.set_xlabel("Ronda global")
        ax.set_ylabel("|acc_A - acc_B|")
        ax.set_xlim(1, MAX_ROUND_FOR_PLOTS)
        ax.legend(fontsize=8, loc="best")
        path = output_dir / "gateway_accuracy_skew.png"
        _save_plot(fig, path)
        plot_paths["gateway_accuracy_skew"] = str(path)

    # 7. class_mix_by_gateway
    if not class_mix.empty:
        share = (
            class_mix.groupby(["variant", "device_suffix", "sample_label_name"])["samples"]
            .sum()
            .reset_index()
        )
        share["variant_gw"] = share["variant"] + " | " + share["device_suffix"]
        pivot = (
            share.pivot(index="variant_gw", columns="sample_label_name", values="samples")
            .fillna(0)
            .sort_index()
        )
        fig, ax = plt.subplots(figsize=(13, 7))
        pivot.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("Distribucion de clases observadas por variante y gateway")
        ax.set_xlabel("Variante | Gateway")
        ax.set_ylabel("Muestras")
        ax.legend(title="Clase", fontsize=8, loc="upper right")
        ax.tick_params(axis="x", rotation=60)
        path = output_dir / "class_mix_by_gateway.png"
        _save_plot(fig, path)
        plot_paths["class_mix_by_gateway"] = str(path)

    return plot_paths


# ---------------------------------------------------------------------------
# SPEC writer
# ---------------------------------------------------------------------------

def write_spec_markdown(
    output_dir: Path,
    executive_summary: dict[str, Any],
    transport_summary: pd.DataFrame,
    local_summary: pd.DataFrame,
    global_summary: pd.DataFrame,
) -> Path:
    path = output_dir / "SRE_OBSERVABILITY_SPEC.md"

    def _fmt(v: Any, decimals: int = 4) -> str:
        if isinstance(v, float):
            if pd.isna(v):
                return "n/a"
            return f"{v:.{decimals}f}"
        return str(v)

    lines = [
        "# SRE Observability Spec - hfl_v7 Consolidado",
        "",
        "Generado a partir de `Results` y `Results_FOG` de las cuatro carpetas de variante",
        "(RN/CNN x ASCON/PLAIN), sin modificar los CSV fuente. Esta spec extiende el SRE",
        "previo de `Analisis de Modelos/RN/` para cubrir las 8 combinaciones experimentales",
        "y permitir comparacion homogenea.",
        "",
        "## Variantes cubiertas",
        "",
        "| Variante | Modelo | Seguridad | Topologia | Carpeta |",
        "| --- | --- | --- | --- | --- |",
    ]
    for spec in VARIANTS:
        lines.append(
            f"| `{spec.label}` | {spec.model} | {spec.security} | {spec.topology} | "
            f"`{spec.results_dir.relative_to(REPO_SRC)}` |"
        )

    lines += [
        "",
        "## Executive Summary global",
        "",
        f"- Variantes con datos: `{executive_summary['global'].get('variants_count', 0)}`",
        f"- Total de intentos analizados: `{executive_summary.get('attempts_analyzed_total', 0)}`",
        f"- Accuracy global final promedio (todas las variantes): `{_fmt(executive_summary['global'].get('avg_global_last_accuracy'))}`",
        f"- Loss global final promedio: `{_fmt(executive_summary['global'].get('avg_global_last_loss'))}`",
        f"- Overhead promedio de payload (bytes): `{_fmt(executive_summary['global'].get('avg_payload_overhead_bytes'), 1)}`",
        f"- Duracion promedio de ronda (s): `{_fmt(executive_summary['global'].get('avg_round_duration_sec'), 2)}`",
        "",
        "## Executive Summary por variante",
        "",
        "| Variante | Intentos | Acc final | Loss final | Round (s) | GW p95 (ms) | Server dec p95 (ms) | Overhead (B) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, payload in sorted(executive_summary.get("variants", {}).items()):
        lines.append(
            f"| `{variant}` | {payload['attempts_analyzed']} | "
            f"{_fmt(payload['avg_global_last_accuracy'])} | "
            f"{_fmt(payload['avg_global_last_loss'])} | "
            f"{_fmt(payload['avg_round_duration_sec'], 2)} | "
            f"{_fmt(payload['avg_gateway_decrypt_p95_ms'], 3)} | "
            f"{_fmt(payload['avg_server_decrypt_p95_ms'], 3)} | "
            f"{_fmt(payload['avg_payload_overhead_bytes'], 1)} |"
        )

    lines += [
        "",
        "## Metric Catalog",
        "",
        "| Metrica | Exactitud | Fuente | Agregacion | Proposito |",
        "| --- | --- | --- | --- | --- |",
        "| `transport.edge_rpi.decrypt.p95_ms` | exact_from_csv | gateway transport | p95(elapsed_ms) WHERE channel='ESP32->RPi' AND operation='decrypt' | SLI de latencia de ingreso al gateway por muestra. |",
        "| `transport.rpi_pc.decrypt.p95_ms` | exact_from_csv | server transport | p95(elapsed_ms) WHERE channel='RPi->PC' AND operation='decrypt' | SLI de descifrado del agregador. |",
        "| `transport.pc_rpi.encrypt.p95_ms` | exact_from_csv | server transport | p95(elapsed_ms) WHERE channel='PC->RPi' AND operation='encrypt' | SLI de despliegue del modelo global. |",
        "| `transport.payload.overhead.avg_bytes` | exact_from_csv | gateway+server transport | avg(overhead_bytes) | Costo promedio del envelope ASCON. |",
        "| `transport.payload.expansion.avg_ratio` | exact_from_csv | gateway+server transport | avg(enc_bytes / pt_bytes) | Factor de expansion del payload. |",
        "| `training.local.accuracy.avg` | exact_from_csv | model_metrics | avg(accuracy) WHERE stage='local_train' | Calidad media del entrenamiento local. |",
        "| `training.local.loss.avg` | exact_from_csv | model_metrics | avg(loss) WHERE stage='local_train' | Estabilidad media del entrenamiento local. |",
        "| `training.gateway.accuracy.skew.avg` | exact_from_csv | model_metrics A/B | avg(\\|acc_A - acc_B\\|) por round | Desalineacion entre gateways. |",
        "| `round.global.accuracy.last` | exact_from_csv | global_weights_history | last(accuracy) por intento | Resultado final del modelo global. |",
        "| `round.global.loss.last` | exact_from_csv | global_weights_history | last(loss) por intento | Punto final de convergencia. |",
        "| `round.duration.avg_sec` | exact_from_csv | global_weights_history | avg(diff(time)) por intento | Duracion promedio de ronda. |",
        "| `round.weight_drift.avg` | exact_from_csv | global_weights_history | avg(\\|delta(w3,w4*)\\|) | Movimiento medio de pesos globales. |",
        "| `reliability.round_completion.rate` | exact_with_assumption | global_weights_history | observed_rounds / expected_rounds | Disponibilidad experimental. |",
        "| `reliability.gateway_participation.rate` | exact_from_csv | model_metrics | local_train_rounds / global_rounds_observed | Tasa efectiva por gateway. |",
        "",
        "## Log Catalog",
        "",
        "| Evento | Estado | Fuente | Campos clave |",
        "| --- | --- | --- | --- |",
        "| `transport.crypto.gateway` | reconstructed_from_csv | ascon_metrics_*gateway / *plain_metrics_gateway | timestamp, variant, attempt_id, gateway_id, channel, operation, elapsed_ms, pt_bytes, enc_bytes, overhead_bytes, client_id, sample_label_name |",
        "| `transport.crypto.server` | reconstructed_from_csv | ascon_metrics_server / plain_metrics_server | timestamp, variant, attempt_id, channel, operation, elapsed_ms, pt_bytes, enc_bytes, overhead_bytes, round_ref |",
        "| `model.local_train` | reconstructed_from_csv | model_metrics_gateway_* | timestamp, variant, attempt_id, gateway_id, fl_round, num_samples, accuracy, loss, buffer_target |",
        "| `model.global_round` | reconstructed_from_csv | global_weights_history_* | timestamp, variant, attempt_id, round_ref, accuracy, loss, w3_mag, w4_normal, w4_brute, w4_scan |",
        "| `fedavg.compute` | live_stdout_only | server_hfl.py | trace_id, ts_start, ts_end, gateways_received, fedavg_ms |",
        "| `model.deploy.gateway` | live_stdout_only | gateway_hfl.py | trace_id, ts, gateway_id, round_ref, payload_bytes, apply_ms |",
        "",
        "## Trace Catalog",
        "",
        "| Traza | Estado | Pattern | Spans |",
        "| --- | --- | --- | --- |",
        "| `round_trace` | reconstructed_from_csv | `variant:{v}:attempt:{a}:round:{r}` | gateway_*.local_train, server.decrypt_batch, server.encrypt_global, global_round_commit |",
        "| `gateway_round_trace` | reconstructed_from_csv | `variant:{v}:attempt:{a}:round:{r}:gateway:{gw}` | esp32_to_rpi_decrypt_batch, local_train_done |",
        "| `sample_trace` | live_stdout_only | `variant:{v}:attempt:{a}:round:{r}:sample:{client_id}:{seq}` | esp32.publish, gateway.decrypt, gateway.buffer, gateway.train_enqueue |",
        "| `fedavg_trace` | live_stdout_only | `variant:{v}:attempt:{a}:round:{r}:fedavg` | server.wait_updates, server.fedavg_compute, server.encrypt_global, server.deploy |",
        "",
        "## Dashboard panels recomendados",
        "",
        "1. **System Health** -- completion rate por variante, gateway participation, last accuracy/loss.",
        "2. **Transport** -- p95 ESP32->RPi/RPi->PC/PC->RPi por variante; overhead bytes y expansion ratio (clave para diferenciar PLAIN vs ASCON).",
        "3. **Local Training** -- accuracy/loss por gateway, skew A vs B, num_samples vs buffer_target.",
        "4. **Global Convergence** -- accuracy/loss por ronda, weight drift por componente.",
        "5. **Data Quality** -- class mix por variante x gateway; deteccion de class imbalance.",
        "",
        "## Alcance",
        "",
        "- **Reconstruible ahora desde CSV**: metricas, eventos canonicos y trazas a nivel de ronda; con eje `variant` se puede comparar RN vs CNN, ASCON vs PLAIN y NoFOG vs FOG en una misma tabla.",
        "- **Solo live stdout**: spans internos de FedAvg en el servidor, aplicacion del modelo en gateway, trazas por muestra individual y latencia real de red extremo a extremo.",
        "",
        "## Diferencias entre variantes para el lector",
        "",
        "- En variantes `*_PLAIN_*` los CSV de transport reportan `payload_bytes` (sin tag/nonce).",
        "  Para mantener la semantica unificada se mapean `serialize/deserialize` a `encrypt/decrypt`",
        "  y se replica `payload_bytes` en `pt_bytes` y `enc_bytes` con `overhead_bytes = 0`.",
        "- En variantes FOG el gateway A puede aparecer como `gateway_fog_leader` (no-ascon) o `gateway_A` (ASCON).",
        "  El eje `device_suffix` conserva el valor original del CSV.",
        "- Los datos PLAIN contienen `operation_raw` con la operacion original (`serialize`/`deserialize`) en `canonical_log_events.csv`.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Cargando datos de {len(VARIANTS)} variantes...")
    data = load_all()
    print(f"[INFO] Inventario:\n{data['attempt_inventory'].groupby('variant').size().to_string()}")

    transport_events = pd.concat(
        [data["gateway_transport"], data["server_transport"]],
        ignore_index=True, sort=False,
    )
    canonical = build_canonical_log_events(data)
    transport_summary = compute_transport_sli(transport_events)
    local_summary = compute_local_training_sli(data["model_metrics"])
    global_summary, global_enriched = compute_global_round_sli(
        data["global_history"], EXPECTED_ROUNDS_PER_ATTEMPT
    )
    alignment = compute_gateway_alignment(data["model_metrics"], data["global_history"])
    class_mix = compute_class_mix(data["gateway_transport"])
    round_trace = build_round_trace_summary(
        data["gateway_transport"],
        data["server_transport"],
        data["model_metrics"],
        data["global_history"],
    )
    executive = compute_executive_summary(
        data["attempt_inventory"], transport_summary, local_summary, global_summary, class_mix
    )
    plot_paths = generate_plots(
        OUTPUT_DIR, transport_summary, alignment, global_enriched, class_mix
    )

    # Salidas exigidas por la tarea
    _save_dataframe(canonical, OUTPUT_DIR / "canonical_log_events.csv")
    _save_dataframe(transport_summary, OUTPUT_DIR / "transport_sli_summary.csv")
    _save_dataframe(local_summary, OUTPUT_DIR / "local_training_sli_summary.csv")
    _save_dataframe(global_summary, OUTPUT_DIR / "global_round_sli_summary.csv")
    _save_dataframe(round_trace, OUTPUT_DIR / "round_trace_summary.csv")
    # Auxiliares utiles
    _save_dataframe(data["attempt_inventory"], OUTPUT_DIR / "attempt_inventory.csv")
    _save_dataframe(global_enriched, OUTPUT_DIR / "global_round_enriched.csv")
    _save_dataframe(alignment, OUTPUT_DIR / "gateway_alignment_sli_summary.csv")
    _save_dataframe(class_mix, OUTPUT_DIR / "class_mix_summary.csv")

    (OUTPUT_DIR / "executive_summary.json").write_text(
        json.dumps(executive, indent=2, default=str), encoding="utf-8"
    )

    spec_path = write_spec_markdown(
        OUTPUT_DIR, executive, transport_summary, local_summary, global_summary
    )

    print(f"[INFO] Listo. Salida en: {OUTPUT_DIR}")
    print(f"[INFO] SPEC: {spec_path}")
    print(f"[INFO] Plots: {list(plot_paths.keys())}")
    return {
        "output_dir": OUTPUT_DIR,
        "executive_summary": executive,
        "plot_paths": plot_paths,
    }


if __name__ == "__main__":
    main()
