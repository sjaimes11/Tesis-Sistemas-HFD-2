"""
=============================================================================
 data_loader.py — Carga y preprocesamiento de datasets IoT-IDS
=============================================================================
 Datasets: uniflow_normal, uniflow_mqtt_bruteforce, uniflow_scan_A
 Features: las mismas 13 que usa el sistema HFL en ESP32
 Clases:   0=normal, 1=mqtt_bruteforce, 2=scan_A
=============================================================================
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "num_pkts", "mean_iat", "std_iat", "min_iat", "max_iat",
    "mean_pkt_len", "num_bytes", "num_psh_flags", "num_rst_flags",
    "num_urg_flags", "std_pkt_len", "min_pkt_len", "max_pkt_len"
]

CLASS_NAMES = ["normal", "mqtt_bruteforce", "scan_A"]

def _default_data_dir() -> Path:
    """
    Devuelve una ruta base razonable tanto en scripts normales como en notebooks.
    En Jupyter `__file__` no existe, así que se usa el directorio actual.
    """
    linux_server_dir = Path("/home/sajaimesp/Tesis_Sistemas/Data_Sets")
    if linux_server_dir.exists():
        return linux_server_dir

    if "__file__" in globals():
        repo_data_dir = Path(__file__).resolve().parents[1] / "Data"
        if repo_data_dir.exists():
            return repo_data_dir

    cwd_data_dir = Path.cwd() / "Data"
    if cwd_data_dir.exists():
        return cwd_data_dir

    return linux_server_dir


def resolve_data_dir(data_dir: Optional[str] = None) -> Path:
    """
    Resuelve el directorio de datasets de forma portable.

    Prioridad:
    1. argumento `data_dir`
    2. variable de entorno `ML_CLASSIC_DATA_DIR`
    3. `<repo>/Data`
    """
    if data_dir:
        return Path(data_dir).expanduser().resolve()

    env_dir = os.environ.get("ML_CLASSIC_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()

    return _default_data_dir()


def load_dataset(
    max_samples_per_class: Optional[int] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
    data_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler]]:
    """
    Carga los 3 CSVs, extrae las 13 features, asigna etiqueta por archivo,
    limita muestras por clase si se indica, divide train/test y escala.

    Returns: X_train, X_test, y_train, y_test, scaler
    """
    dataset_dir = resolve_data_dir(data_dir)
    files = {
        0: dataset_dir / "uniflow_normal.csv",
        1: dataset_dir / "uniflow_mqtt_bruteforce.csv",
        2: dataset_dir / "uniflow_scan_A.csv",
    }

    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        joined = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "No se encontraron todos los CSV requeridos. "
            "Ajusta `--data-dir` o la variable `ML_CLASSIC_DATA_DIR`.\n"
            f"{joined}"
        )

    frames = []
    for label, path in files.items():
        df = pd.read_csv(path)
        df = df[FEATURE_COLUMNS].copy()
        df["label"] = label

        if max_samples_per_class and len(df) > max_samples_per_class:
            df = df.sample(n=max_samples_per_class, random_state=random_state)

        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    X = data[FEATURE_COLUMNS].values.astype(np.float32)
    y = data["label"].values.astype(np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    print(f"\nDatasets cargados desde: {dataset_dir}")
    print(f"Dataset cargado: {len(X_train)} train, {len(X_test)} test")
    for i, name in enumerate(CLASS_NAMES):
        n_train = np.sum(y_train == i)
        n_test = np.sum(y_test == i)
        print(f"  Clase {i} ({name:>18}): {n_train:>7} train, {n_test:>6} test")

    class_counts = np.bincount(y_train)
    max_count = class_counts.max()
    ratios = max_count / class_counts
    print(f"  Ratio desbalance (vs mayor): {' / '.join(f'{r:.1f}' for r in ratios)}")
    print(f"  -> Se recomienda class_weight='balanced' en todos los modelos\n")

    return X_train, X_test, y_train, y_test, scaler


def print_evaluation(model_name, y_test, y_pred, best_params=None, cv_scores=None):
    """Imprime métricas de evaluación completas para tesis."""
    from sklearn.metrics import (
        classification_report, confusion_matrix, accuracy_score,
        f1_score, precision_score, recall_score, matthews_corrcoef
    )

    y_test = np.asarray(y_test, dtype=np.int32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int32).reshape(-1)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    prec_macro = precision_score(y_test, y_pred, average="macro")
    rec_macro = recall_score(y_test, y_pred, average="macro")
    mcc = matthews_corrcoef(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f" {model_name}")
    print(f"{'='*60}")
    if best_params:
        print(f" Mejores hiperparámetros: {best_params}")
    if cv_scores is not None:
        print(f" CV F1 macro:   {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print(f" Accuracy:      {acc:.4f}")
    print(f" F1 macro:      {f1_macro:.4f}")
    print(f" F1 weighted:   {f1_weighted:.4f}")
    print(f" Precision:     {prec_macro:.4f}")
    print(f" Recall:        {rec_macro:.4f}")
    print(f" MCC:           {mcc:.4f}")
    print(f"\n Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))
    print(" Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  {'':>15} {'normal':>10} {'bruteforce':>10} {'scan_A':>10}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:>15} {cm[i][0]:>10} {cm[i][1]:>10} {cm[i][2]:>10}")
    print()

    return {
        "accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted,
        "precision": prec_macro, "recall": rec_macro, "mcc": mcc
    }
