"""
NIST relevance checks for the HFL v7 experiments.

This script uses the already captured CSV files and does not modify raw
experiment results. It produces evidence tables for:
- ASCON operation counts and operational thresholds.
- Attempt-level statistical checks for ASCON vs no-ASCON.
- Thesis-ready notes separating NIST facts from project assumptions.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-darkgrid")
warnings.filterwarnings(
    "ignore",
    message="DataFrameGroupBy.apply operated on the grouping columns",
    category=DeprecationWarning,
)


ROOT = Path(r"C:\Users\VivoBook\Downloads\microproyecto2\microproyecto 2 porque wtf\src")
OUTPUT_ROOT = ROOT / "Analisis de Modelos" / "NIST" / "analysis_outputs" / "nist_relevance"

ALPHA = 0.05
PROJECT_MIN_ASCON_OPS = 1000
PROJECT_LATENCY_P95_LIMIT_MS = 50.0
PROJECT_OVERHEAD_LIMIT_PCT = 50.0
RECOMMENDED_INDEPENDENT_RUNS = 30
EXPECTED_ROUNDS_PER_ATTEMPT = 30
ATTEMPT_RE = re.compile(r"#(\d+)$")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    architecture: str
    mode: str
    results_root: Path
    attempt_start: int
    attempt_end: int
    crypto: str
    fog: bool = False


EXPERIMENTS = [
    ExperimentConfig(
        name="RN_ASCON",
        architecture="RN",
        mode="ascon",
        results_root=ROOT / "hfl_v7-RN" / "Results",
        attempt_start=3,
        attempt_end=11,
        crypto="ASCON",
    ),
    ExperimentConfig(
        name="CNN_FOG_ASCON",
        architecture="CNN_FOG",
        mode="ascon",
        results_root=ROOT / "hfl_v7-CNN" / "Results_FOG",
        attempt_start=2,
        attempt_end=8,
        crypto="ASCON",
        fog=True,
    ),
    ExperimentConfig(
        name="RN_NO_ASCON",
        architecture="RN",
        mode="plain",
        results_root=ROOT / "hfl_v7-no-ascon" / "Results",
        attempt_start=16,
        attempt_end=29,
        crypto="NO_ASCON",
    ),
]


def _safe_mean(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.mean()) if not clean.empty else float("nan")


def _safe_std(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.std(ddof=1)) if len(clean) > 1 else float("nan")


def _safe_pct(series: pd.Series, q: float) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(np.percentile(clean.to_numpy(dtype=float), q)) if not clean.empty else float("nan")


def _discover_attempt_dirs(config: ExperimentConfig) -> list[tuple[int, Path]]:
    attempts: list[tuple[int, Path]] = []
    if not config.results_root.exists():
        return attempts
    for path in config.results_root.iterdir():
        if not path.is_dir():
            continue
        match = ATTEMPT_RE.match(path.name)
        if not match:
            continue
        attempt_id = int(match.group(1))
        if config.attempt_start <= attempt_id <= config.attempt_end:
            attempts.append((attempt_id, path))
    return sorted(attempts, key=lambda item: item[0])


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _first_existing(attempt_dir: Path, pattern: str) -> Path | None:
    matches = sorted(attempt_dir.glob(pattern))
    return matches[0] if matches else None


def _normalize_channel(channel: Any) -> str:
    return str(channel).replace("RPi_leader->PC", "RPi->PC").replace("PC->RPi_leader", "PC->RPi")


def load_transport(config: ExperimentConfig) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for attempt_id, attempt_dir in _discover_attempt_dirs(config):
        if config.mode == "ascon":
            paths = sorted(attempt_dir.glob("ascon_metrics_*.csv"))
        else:
            paths = sorted(attempt_dir.glob("plain_metrics_*.csv"))

        for path in paths:
            df = _read_csv(path)
            if df.empty:
                continue
            df["experiment"] = config.name
            df["architecture"] = config.architecture
            df["mode"] = config.mode
            df["crypto"] = config.crypto
            df["attempt_id"] = attempt_id
            df["source_file"] = path.name
            df["source_scope"] = "server_transport" if "server" in path.name.lower() else "gateway_transport"
            df["channel"] = df["channel"].map(_normalize_channel)
            rows.append(df)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def load_model_metrics(config: ExperimentConfig) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for attempt_id, attempt_dir in _discover_attempt_dirs(config):
        for path in sorted(attempt_dir.glob("model_metrics_gateway_*.csv")):
            df = _read_csv(path)
            if df.empty:
                continue
            df["experiment"] = config.name
            df["architecture"] = config.architecture
            df["mode"] = config.mode
            df["attempt_id"] = attempt_id
            df["source_file"] = path.name
            rows.append(df)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def load_global_history(config: ExperimentConfig) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for attempt_id, attempt_dir in _discover_attempt_dirs(config):
        path = _first_existing(attempt_dir, "global_weights_history_*.csv")
        if not path:
            continue
        df = _read_csv(path)
        if df.empty:
            continue
        df["experiment"] = config.name
        df["architecture"] = config.architecture
        df["mode"] = config.mode
        df["attempt_id"] = attempt_id
        df["source_file"] = path.name
        rows.append(df)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def build_attempt_inventory(configs: list[ExperimentConfig]) -> pd.DataFrame:
    rows = []
    for config in configs:
        for attempt_id, attempt_dir in _discover_attempt_dirs(config):
            rows.append(
                {
                    "experiment": config.name,
                    "architecture": config.architecture,
                    "mode": config.mode,
                    "attempt_id": attempt_id,
                    "attempt_dir": str(attempt_dir),
                    "ascon_metric_files": len(list(attempt_dir.glob("ascon_metrics_*.csv"))),
                    "plain_metric_files": len(list(attempt_dir.glob("plain_metrics_*.csv"))),
                    "model_metric_files": len(list(attempt_dir.glob("model_metrics_gateway_*.csv"))),
                    "global_history_files": len(list(attempt_dir.glob("global_weights_history_*.csv"))),
                }
            )
    return pd.DataFrame(rows)


def summarize_ascon_thresholds(transport: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ascon = transport.loc[transport["mode"] == "ascon"].copy()
    if ascon.empty:
        return pd.DataFrame(), pd.DataFrame()

    ascon["elapsed_ms"] = pd.to_numeric(ascon["elapsed_ms"], errors="coerce")
    ascon["pt_bytes"] = pd.to_numeric(ascon["pt_bytes"], errors="coerce")
    ascon["enc_bytes"] = pd.to_numeric(ascon["enc_bytes"], errors="coerce")
    ascon["overhead_bytes"] = pd.to_numeric(ascon["overhead_bytes"], errors="coerce")
    ascon["overhead_pct"] = ascon["overhead_bytes"] / ascon["pt_bytes"] * 100.0
    ascon["expansion_ratio"] = ascon["enc_bytes"] / ascon["pt_bytes"]

    by_attempt = (
        ascon.groupby(["experiment", "architecture", "attempt_id"], dropna=False)
        .apply(
            lambda group: pd.Series(
                {
                    "ascon_ops": int(group["operation"].isin(["encrypt", "decrypt"]).sum()),
                    "encrypt_ops": int((group["operation"] == "encrypt").sum()),
                    "decrypt_ops": int((group["operation"] == "decrypt").sum()),
                    "p50_latency_ms": _safe_pct(group["elapsed_ms"], 50),
                    "p95_latency_ms": _safe_pct(group["elapsed_ms"], 95),
                    "max_latency_ms": pd.to_numeric(group["elapsed_ms"], errors="coerce").max(),
                    "avg_pt_bytes": _safe_mean(group["pt_bytes"]),
                    "avg_enc_bytes": _safe_mean(group["enc_bytes"]),
                    "avg_overhead_bytes": _safe_mean(group["overhead_bytes"]),
                    "avg_overhead_pct": _safe_mean(group["overhead_pct"]),
                    "avg_expansion_ratio": _safe_mean(group["expansion_ratio"]),
                }
            )
        )
        .reset_index()
    )

    by_attempt["passes_min_ops_1000"] = by_attempt["ascon_ops"] >= PROJECT_MIN_ASCON_OPS
    by_attempt["passes_p95_latency_50ms"] = by_attempt["p95_latency_ms"] <= PROJECT_LATENCY_P95_LIMIT_MS
    by_attempt["passes_avg_overhead_50pct"] = by_attempt["avg_overhead_pct"] <= PROJECT_OVERHEAD_LIMIT_PCT

    by_experiment = (
        by_attempt.groupby(["experiment", "architecture"], dropna=False)
        .apply(
            lambda group: pd.Series(
                {
                    "attempts": int(group["attempt_id"].nunique()),
                    "total_ascon_ops": int(group["ascon_ops"].sum()),
                    "min_ops_per_attempt": int(group["ascon_ops"].min()) if not group.empty else 0,
                    "attempts_passing_1000_ops": int(group["passes_min_ops_1000"].sum()),
                    "mean_p95_latency_ms": _safe_mean(group["p95_latency_ms"]),
                    "max_p95_latency_ms": pd.to_numeric(group["p95_latency_ms"], errors="coerce").max(),
                    "mean_overhead_pct": _safe_mean(group["avg_overhead_pct"]),
                    "mean_expansion_ratio": _safe_mean(group["avg_expansion_ratio"]),
                }
            )
        )
        .reset_index()
    )
    by_experiment["passes_total_ops_1000"] = by_experiment["total_ascon_ops"] >= PROJECT_MIN_ASCON_OPS
    by_experiment["passes_all_attempts_1000_ops"] = by_experiment["attempts_passing_1000_ops"] == by_experiment["attempts"]
    by_experiment["passes_p95_latency_50ms"] = by_experiment["max_p95_latency_ms"] <= PROJECT_LATENCY_P95_LIMIT_MS
    by_experiment["passes_avg_overhead_50pct"] = by_experiment["mean_overhead_pct"] <= PROJECT_OVERHEAD_LIMIT_PCT
    return by_attempt, by_experiment


def summarize_independent_runs(inventory: pd.DataFrame, global_history: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for experiment, group in inventory.groupby("experiment"):
        gh = global_history.loc[global_history["experiment"] == experiment]
        rounds = int(pd.to_numeric(gh.get("round"), errors="coerce").dropna().count()) if not gh.empty else 0
        rows.append(
            {
                "experiment": experiment,
                "independent_attempts": int(group["attempt_id"].nunique()),
                "observed_round_rows": rounds,
                "recommended_independent_attempts": RECOMMENDED_INDEPENDENT_RUNS,
                "passes_30_independent_attempts": int(group["attempt_id"].nunique()) >= RECOMMENDED_INDEPENDENT_RUNS,
                "round_rows_at_least_30": rounds >= RECOMMENDED_INDEPENDENT_RUNS,
                "note": "Round rows are repeated measures, not fully independent experimental executions.",
            }
        )
    return pd.DataFrame(rows).sort_values("experiment").reset_index(drop=True)


def build_attempt_level_metrics(transport: pd.DataFrame, model_metrics: pd.DataFrame, global_history: pd.DataFrame) -> pd.DataFrame:
    records = []
    keys = sorted(set(transport["experiment"].dropna().unique()) | set(global_history["experiment"].dropna().unique()))
    for experiment in keys:
        attempts = sorted(
            set(transport.loc[transport["experiment"] == experiment, "attempt_id"].dropna().astype(int).unique())
            | set(global_history.loc[global_history["experiment"] == experiment, "attempt_id"].dropna().astype(int).unique())
        )
        for attempt_id in attempts:
            t = transport.loc[(transport["experiment"] == experiment) & (transport["attempt_id"] == attempt_id)].copy()
            m = model_metrics.loc[(model_metrics["experiment"] == experiment) & (model_metrics["attempt_id"] == attempt_id)].copy()
            g = global_history.loc[(global_history["experiment"] == experiment) & (global_history["attempt_id"] == attempt_id)].copy()
            mode = t["mode"].dropna().iloc[0] if not t.empty else (g["mode"].dropna().iloc[0] if not g.empty else "")
            architecture = t["architecture"].dropna().iloc[0] if not t.empty else (g["architecture"].dropna().iloc[0] if not g.empty else "")

            if mode == "ascon":
                edge_op = "decrypt"
                server_in_op = "decrypt"
                server_out_op = "encrypt"
                edge_size_col = "enc_bytes"
            else:
                edge_op = "deserialize"
                server_in_op = "deserialize"
                server_out_op = "serialize"
                edge_size_col = "payload_bytes"

            edge = t.loc[
                (t["source_scope"] == "gateway_transport")
                & (t["channel"] == "ESP32->RPi")
                & (t["operation"] == edge_op)
            ]
            server_in = t.loc[
                (t["source_scope"] == "server_transport")
                & (t["channel"] == "RPi->PC")
                & (t["operation"] == server_in_op)
            ]
            server_out = t.loc[
                (t["source_scope"] == "server_transport")
                & (t["channel"] == "PC->RPi")
                & (t["operation"] == server_out_op)
            ]

            g = g.sort_values("round") if not g.empty and "round" in g.columns else g
            local = m.loc[m.get("stage", pd.Series(dtype=str)).astype(str) == "local_train"] if not m.empty else pd.DataFrame()
            fog = m.loc[m.get("stage", pd.Series(dtype=str)).astype(str) == "fog_fedavg"] if not m.empty else pd.DataFrame()

            record = {
                "experiment": experiment,
                "architecture": architecture,
                "mode": mode,
                "attempt_id": attempt_id,
                "edge_processing_p95_ms": _safe_pct(edge["elapsed_ms"], 95),
                "server_ingress_processing_p95_ms": _safe_pct(server_in["elapsed_ms"], 95),
                "server_egress_processing_p95_ms": _safe_pct(server_out["elapsed_ms"], 95),
                "edge_payload_bytes_avg": _safe_mean(edge[edge_size_col]) if edge_size_col in edge.columns else float("nan"),
                "transport_events": int(len(t)),
                "ascon_ops": int(t["operation"].isin(["encrypt", "decrypt"]).sum()) if mode == "ascon" and "operation" in t else 0,
                "avg_local_accuracy": _safe_mean(local["accuracy"]) if not local.empty else float("nan"),
                "avg_local_loss": _safe_mean(local["loss"]) if not local.empty else float("nan"),
                "avg_fog_accuracy": _safe_mean(fog["accuracy"]) if not fog.empty else float("nan"),
                "avg_fog_loss": _safe_mean(fog["loss"]) if not fog.empty else float("nan"),
                "rounds_observed": int(g["round"].nunique()) if not g.empty and "round" in g.columns else 0,
                "round_completion_rate": float(g["round"].nunique() / EXPECTED_ROUNDS_PER_ATTEMPT) if not g.empty and "round" in g.columns else float("nan"),
                "last_global_accuracy": pd.to_numeric(g["accuracy"], errors="coerce").iloc[-1] if not g.empty and "accuracy" in g.columns else float("nan"),
                "last_global_loss": pd.to_numeric(g["loss"], errors="coerce").iloc[-1] if not g.empty and "loss" in g.columns else float("nan"),
                "avg_round_duration_sec": _mean_round_duration(g),
            }
            if mode == "ascon" and not t.empty and {"overhead_bytes", "pt_bytes"}.issubset(t.columns):
                overhead_pct = pd.to_numeric(t["overhead_bytes"], errors="coerce") / pd.to_numeric(t["pt_bytes"], errors="coerce") * 100.0
                record["avg_overhead_pct"] = _safe_mean(overhead_pct)
                record["avg_overhead_bytes"] = _safe_mean(t["overhead_bytes"])
            else:
                record["avg_overhead_pct"] = 0.0 if mode == "plain" else float("nan")
                record["avg_overhead_bytes"] = 0.0 if mode == "plain" else float("nan")
            records.append(record)
    return pd.DataFrame(records).sort_values(["experiment", "attempt_id"]).reset_index(drop=True)


def _mean_round_duration(global_attempt: pd.DataFrame) -> float:
    if global_attempt.empty or "time" not in global_attempt.columns:
        return float("nan")
    times = pd.to_datetime("2026-01-01 " + global_attempt["time"].astype(str), errors="coerce")
    durations = times.sort_values().diff().dt.total_seconds()
    return _safe_mean(durations)


def _bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, repeats: int = 5000, seed: int = 42) -> tuple[float, float]:
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(repeats):
        sample_a = rng.choice(a, size=len(a), replace=True)
        sample_b = rng.choice(b, size=len(b), replace=True)
        diffs.append(float(np.mean(sample_a) - np.mean(sample_b)))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled = math.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else float("nan")


def compare_rn_ascon_vs_no_ascon(attempt_metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "edge_processing_p95_ms",
        "server_ingress_processing_p95_ms",
        "server_egress_processing_p95_ms",
        "edge_payload_bytes_avg",
        "avg_round_duration_sec",
        "last_global_accuracy",
        "last_global_loss",
        "round_completion_rate",
    ]
    rows = []
    a_group = attempt_metrics.loc[attempt_metrics["experiment"] == "RN_ASCON"]
    b_group = attempt_metrics.loc[attempt_metrics["experiment"] == "RN_NO_ASCON"]

    for metric in metrics:
        a = pd.to_numeric(a_group[metric], errors="coerce").dropna().to_numpy(dtype=float)
        b = pd.to_numeric(b_group[metric], errors="coerce").dropna().to_numpy(dtype=float)
        if len(a) < 2 or len(b) < 2:
            continue

        shapiro_a = stats.shapiro(a).pvalue if 3 <= len(a) <= 5000 and np.unique(a).size > 1 else float("nan")
        shapiro_b = stats.shapiro(b).pvalue if 3 <= len(b) <= 5000 and np.unique(b).size > 1 else float("nan")
        normal_a = bool(shapiro_a >= ALPHA) if not math.isnan(shapiro_a) else False
        normal_b = bool(shapiro_b >= ALPHA) if not math.isnan(shapiro_b) else False

        if normal_a and normal_b:
            test_name = "welch_ttest"
            stat, p_value = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        else:
            test_name = "mann_whitney_u"
            stat, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")

        mean_a = float(np.mean(a))
        mean_b = float(np.mean(b))
        ci_low, ci_high = _bootstrap_mean_diff(a, b)
        rows.append(
            {
                "comparison": "RN_ASCON_vs_RN_NO_ASCON",
                "metric": metric,
                "n_ascon": int(len(a)),
                "n_no_ascon": int(len(b)),
                "mean_ascon": mean_a,
                "mean_no_ascon": mean_b,
                "mean_diff_ascon_minus_no_ascon": mean_a - mean_b,
                "percent_change_vs_no_ascon": ((mean_a - mean_b) / mean_b * 100.0) if mean_b != 0 else float("nan"),
                "ci95_mean_diff_low": ci_low,
                "ci95_mean_diff_high": ci_high,
                "shapiro_p_ascon": shapiro_a,
                "shapiro_p_no_ascon": shapiro_b,
                "selected_test": test_name,
                "test_statistic": float(stat),
                "p_value": float(p_value),
                "alpha": ALPHA,
                "statistically_significant": bool(p_value < ALPHA),
                "cohens_d": _cohens_d(a, b),
                "paired_design": False,
                "note": "Mann-Whitney is used for non-normal independent samples; Wilcoxon signed-rank would require paired runs.",
            }
        )
    return pd.DataFrame(rows)


def build_claim_audit(ascon_experiment_summary: pd.DataFrame, independent_summary: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "claim": "ASCON was selected by NIST for lightweight cryptography standardization.",
            "status": "supported_by_nist",
            "evidence": "NIST selected the Ascon family in 2023 and later finalized an Ascon-based lightweight cryptography standard.",
            "source": "NIST LWC / SP 800-232",
        },
        {
            "claim": "SP 800-22 validates n>=30 FL experiment runs.",
            "status": "needs_correction",
            "evidence": "SP 800-22 is a statistical test suite for random and pseudorandom number generators, not a guideline for FL accuracy/loss experiments.",
            "source": "NIST SP 800-22",
        },
        {
            "claim": "At least 1000 ASCON operations were observed.",
            "status": "data_supported" if (ascon_experiment_summary["total_ascon_ops"] >= PROJECT_MIN_ASCON_OPS).any() else "not_supported",
            "evidence": "See ascon_threshold_summary_by_experiment.csv.",
            "source": "project CSV metrics",
        },
        {
            "claim": "At least 30 independent full experiment executions were captured.",
            "status": "data_supported"
            if independent_summary["passes_30_independent_attempts"].fillna(False).any()
            else "not_supported",
            "evidence": "Current datasets have fewer than 30 independent attempts per experiment. Round rows can be analyzed as repeated measures, not independent executions.",
            "source": "project CSV metrics",
        },
        {
            "claim": "Average ASCON overhead stays below 50%.",
            "status": "data_supported"
            if ascon_experiment_summary["passes_avg_overhead_50pct"].fillna(False).all()
            else "not_supported_or_partial",
            "evidence": "Small JSON payloads plus base64 envelope can exceed 50% overhead. See threshold tables.",
            "source": "project CSV metrics",
        },
    ]
    return pd.DataFrame(rows)


def generate_plots(output_dir: Path, ascon_by_experiment: pd.DataFrame, attempt_metrics: pd.DataFrame, stats_table: pd.DataFrame) -> dict[str, str]:
    plots: dict[str, str] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ascon_by_experiment.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(ascon_by_experiment["experiment"], ascon_by_experiment["total_ascon_ops"], color="#1f77b4")
        ax.axhline(PROJECT_MIN_ASCON_OPS, color="#d62728", linestyle="--", label=f"{PROJECT_MIN_ASCON_OPS} ops")
        ax.set_title("ASCON operations observed by experiment")
        ax.set_ylabel("encrypt/decrypt operations")
        ax.tick_params(axis="x", rotation=20)
        ax.legend()
        path = output_dir / "ascon_operations_threshold.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        plots["ascon_operations_threshold"] = str(path)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(ascon_by_experiment["experiment"], ascon_by_experiment["mean_overhead_pct"], color="#ff7f0e")
        ax.axhline(PROJECT_OVERHEAD_LIMIT_PCT, color="#d62728", linestyle="--", label="50% project threshold")
        ax.set_title("Average ASCON payload overhead percentage")
        ax.set_ylabel("overhead (%)")
        ax.tick_params(axis="x", rotation=20)
        ax.legend()
        path = output_dir / "ascon_overhead_threshold.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        plots["ascon_overhead_threshold"] = str(path)

    if not attempt_metrics.empty:
        subset = attempt_metrics.loc[attempt_metrics["experiment"].isin(["RN_ASCON", "RN_NO_ASCON"])].copy()
        if not subset.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            subset.boxplot(column="edge_processing_p95_ms", by="experiment", ax=ax)
            ax.set_title("Gateway edge processing p95: ASCON vs no-ASCON")
            ax.set_xlabel("experiment")
            ax.set_ylabel("p95 ms")
            plt.suptitle("")
            path = output_dir / "rn_ascon_vs_no_ascon_edge_p95.png"
            fig.tight_layout()
            fig.savefig(path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            plots["rn_ascon_vs_no_ascon_edge_p95"] = str(path)

    if not stats_table.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        chart = stats_table.copy()
        chart["metric_short"] = chart["metric"].str.replace("_", "\n")
        ax.bar(chart["metric_short"], chart["percent_change_vs_no_ascon"], color="#9467bd")
        ax.axhline(0, color="#333333", linewidth=1)
        ax.set_title("Percent change: RN_ASCON vs RN_NO_ASCON")
        ax.set_ylabel("% change vs no-ASCON")
        ax.tick_params(axis="x", rotation=45)
        path = output_dir / "rn_ascon_vs_no_ascon_percent_change.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        plots["rn_ascon_vs_no_ascon_percent_change"] = str(path)

    return plots


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data._"

    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: "" if pd.isna(value) else f"{value:.6g}")
        else:
            view[col] = view[col].map(lambda value: "" if pd.isna(value) else str(value))

    headers = [str(col) for col in view.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in view.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in view.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown_report(
    output_dir: Path,
    ascon_summary: pd.DataFrame,
    independent_summary: pd.DataFrame,
    stats_table: pd.DataFrame,
    claim_audit: pd.DataFrame,
    plots: dict[str, str],
) -> Path:
    path = output_dir / "NIST_RELEVANCE_REPORT.md"
    lines = [
        "# NIST Relevance Report - HFL v7",
        "",
        "This report is generated from the project CSV results. It separates NIST-backed claims from project-specific benchmark thresholds.",
        "",
        "## NIST Scope Correction",
        "",
        "- NIST selected the Ascon family for lightweight cryptography standardization for constrained devices in 2023.",
        "- NIST finalized the Ascon-based lightweight cryptography standard in 2025. Present this as selected in 2023 and standardized/finalized later, not as fully finalized in 2023.",
        "- NIST SP 800-22 is for randomness tests of RNG/PRNG bitstreams. It should not be cited as the source for t-tests, Wilcoxon tests, or the n>=30 rule for FL accuracy/loss experiments.",
        "- The `1000 operations`, `p95 <= 50 ms`, and `overhead <= 50%` checks in this folder are project benchmark thresholds, not literal NIST pass/fail requirements.",
        "",
        "## ASCON Threshold Summary",
        "",
        _df_to_markdown(ascon_summary) if not ascon_summary.empty else "_No ASCON data found._",
        "",
        "## Independent Run Sufficiency",
        "",
        _df_to_markdown(independent_summary) if not independent_summary.empty else "_No inventory data found._",
        "",
        "## RN ASCON vs no-ASCON Statistical Tests",
        "",
        _df_to_markdown(stats_table) if not stats_table.empty else "_No comparable attempt-level metrics found._",
        "",
        "## Claim Audit",
        "",
        _df_to_markdown(claim_audit) if not claim_audit.empty else "_No claims audited._",
        "",
        "## Figures",
        "",
    ]
    for name, fig_path in plots.items():
        lines.append(f"- `{name}`: `{fig_path}`")
    lines.extend(
        [
            "",
            "## Official NIST References",
            "",
            "- NIST LWC selection of Ascon, 2023: https://www.nist.gov/news-events/news/2023/02/lightweight-cryptography-standardization-process-nist-selects-ascon",
            "- NIST article on Ascon for small devices, 2023: https://www.nist.gov/news-events/news/2023/02/nist-selects-lightweight-cryptography-algorithms-protect-small-devices",
            "- NIST final lightweight cryptography standard announcement, 2025: https://www.nist.gov/news-events/news/2025/08/nist-finalizes-lightweight-cryptography-standard-protect-small-devices",
            "- NIST SP 800-22 randomness test suite: https://www.nist.gov/publications/statistical-test-suite-random-and-pseudorandom-number-generators-cryptographic-1",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_analysis() -> dict[str, Any]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    inventory = build_attempt_inventory(EXPERIMENTS)
    transport = pd.concat([load_transport(config) for config in EXPERIMENTS], ignore_index=True, sort=False)
    model_metrics = pd.concat([load_model_metrics(config) for config in EXPERIMENTS], ignore_index=True, sort=False)
    global_history = pd.concat([load_global_history(config) for config in EXPERIMENTS], ignore_index=True, sort=False)

    ascon_by_attempt, ascon_by_experiment = summarize_ascon_thresholds(transport)
    independent_summary = summarize_independent_runs(inventory, global_history)
    attempt_metrics = build_attempt_level_metrics(transport, model_metrics, global_history)
    stats_table = compare_rn_ascon_vs_no_ascon(attempt_metrics)
    claim_audit = build_claim_audit(ascon_by_experiment, independent_summary)
    plots = generate_plots(OUTPUT_ROOT, ascon_by_experiment, attempt_metrics, stats_table)

    tables = {
        "attempt_inventory": inventory,
        "transport_events_combined": transport,
        "model_metrics_combined": model_metrics,
        "global_history_combined": global_history,
        "ascon_threshold_summary_by_attempt": ascon_by_attempt,
        "ascon_threshold_summary_by_experiment": ascon_by_experiment,
        "independent_run_sufficiency": independent_summary,
        "attempt_level_metrics": attempt_metrics,
        "rn_ascon_vs_no_ascon_stats": stats_table,
        "nist_claim_audit": claim_audit,
    }
    for name, table in tables.items():
        table.to_csv(OUTPUT_ROOT / f"{name}.csv", index=False, encoding="utf-8")

    executive_summary = {
        "output_dir": str(OUTPUT_ROOT),
        "experiments": [config.name for config in EXPERIMENTS],
        "alpha": ALPHA,
        "project_min_ascon_ops": PROJECT_MIN_ASCON_OPS,
        "project_latency_p95_limit_ms": PROJECT_LATENCY_P95_LIMIT_MS,
        "project_overhead_limit_pct": PROJECT_OVERHEAD_LIMIT_PCT,
        "recommended_independent_runs": RECOMMENDED_INDEPENDENT_RUNS,
        "ascon_experiment_summary": ascon_by_experiment.to_dict(orient="records"),
        "independent_run_summary": independent_summary.to_dict(orient="records"),
        "statistical_tests": stats_table.to_dict(orient="records"),
        "claim_audit": claim_audit.to_dict(orient="records"),
        "plots": plots,
    }
    (OUTPUT_ROOT / "executive_summary.json").write_text(json.dumps(executive_summary, indent=2), encoding="utf-8")
    report_path = write_markdown_report(OUTPUT_ROOT, ascon_by_experiment, independent_summary, stats_table, claim_audit, plots)
    return {
        "output_dir": OUTPUT_ROOT,
        "tables": tables,
        "executive_summary": executive_summary,
        "report_path": report_path,
    }


if __name__ == "__main__":
    bundle = run_analysis()
    print(f"NIST relevance analysis generated in: {bundle['output_dir']}")
