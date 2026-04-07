"""
=============================================================================
 training_runtime.py — Utilidades compartidas para entrenamiento portable
=============================================================================
 Permite:
 - elegir backend CPU/GPU (`scikit-learn` / `cuML`) con fallback controlado
 - ejecutar búsqueda manual de hiperparámetros cuando el backend GPU no usa
   `GridSearchCV` directamente
 - guardar artefactos en rutas configurables
=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid, ParameterSampler


@dataclass(frozen=True)
class BackendInfo:
    requested: str
    selected: str
    rapids_available: bool
    reason: str = ""


@dataclass
class SearchResult:
    best_estimator_: Any
    best_params_: dict[str, Any]
    best_index_: int
    cv_results_: dict[str, np.ndarray]


def get_default_parallel_jobs(selected_backend: str, requested_jobs: int) -> int:
    """Evita sobrecargar la GPU con búsquedas paralelas."""
    if selected_backend == "gpu":
        return 1
    return requested_jobs


def resolve_backend(requested: str = "auto") -> BackendInfo:
    requested = requested.lower()
    if requested not in {"auto", "cpu", "gpu"}:
        raise ValueError(f"Backend inválido: {requested}")

    rapids_error = None
    try:
        import cuml  # noqa: F401
    except Exception as exc:  # pragma: no cover - depende del entorno
        rapids_error = exc

    rapids_available = rapids_error is None

    if requested == "cpu":
        return BackendInfo(
            requested=requested,
            selected="cpu",
            rapids_available=rapids_available,
            reason="Backend CPU solicitado por el usuario.",
        )

    if requested == "gpu":
        if not rapids_available:
            raise RuntimeError(
                "Se solicitó backend GPU pero cuML/RAPIDS no está disponible. "
                "Instala las dependencias GPU en el servidor Linux o usa --backend cpu."
            ) from rapids_error
        return BackendInfo(
            requested=requested,
            selected="gpu",
            rapids_available=True,
            reason="Backend GPU solicitado por el usuario.",
        )

    if rapids_available:
        return BackendInfo(
            requested=requested,
            selected="gpu",
            rapids_available=True,
            reason="cuML/RAPIDS detectado; se usará aceleración GPU.",
        )

    return BackendInfo(
        requested=requested,
        selected="cpu",
        rapids_available=False,
        reason="cuML/RAPIDS no está disponible; se usa scikit-learn en CPU.",
    )


def ensure_output_dir(path_like: str | Path) -> Path:
    output_dir = Path(path_like).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def output_path(output_dir: str | Path, filename: str) -> Path:
    return ensure_output_dir(output_dir) / filename


def dump_model_artifacts(
    model: Any,
    model_path: str | Path,
    scaler: Any = None,
    scaler_path: str | Path | None = None,
) -> list[Path]:
    """
    Guarda el modelo nativo y, cuando cuML lo permite, una copia sklearn.
    Eso facilita `m2cgen` y exportes en máquinas sin GPU.
    """
    saved_paths: list[Path] = []
    model_path = Path(model_path).expanduser().resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    saved_paths.append(model_path)

    if hasattr(model, "as_sklearn"):
        sklearn_path = model_path.with_name(f"{model_path.stem}_sklearn.pkl")
        sklearn_model = model.as_sklearn()
        joblib.dump(sklearn_model, sklearn_path)
        saved_paths.append(sklearn_path)

    if scaler is not None and scaler_path is not None:
        scaler_path = Path(scaler_path).expanduser().resolve()
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        saved_paths.append(scaler_path)

    return saved_paths


def save_feature_order(output_dir: str | Path, feature_names: list[str]) -> Path:
    path = output_path(output_dir, "feature_order.csv")
    pd.DataFrame({"feature": feature_names}).to_csv(path, index=False)
    return path


def save_label_map(output_dir: str | Path, class_names: list[str]) -> Path:
    path = output_path(output_dir, "label_map.json")
    label_map = {str(index): name for index, name in enumerate(class_names)}
    path.write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    return path


def save_scaler_params(output_dir: str | Path, scaler: Any) -> Optional[Path]:
    if scaler is None:
        return None

    path = output_path(output_dir, "scaler_params.json")
    payload = {
        "mean": [float(value) for value in scaler.mean_],
        "scale": [float(value) for value in scaler.scale_],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def save_edge_metadata(
    output_dir: str | Path,
    feature_names: list[str],
    class_names: list[str],
    scaler: Any = None,
) -> list[Path]:
    saved = [
        save_feature_order(output_dir, feature_names),
        save_label_map(output_dir, class_names),
    ]
    scaler_path = save_scaler_params(output_dir, scaler)
    if scaler_path is not None:
        saved.append(scaler_path)
    return saved


def to_numpy(data: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Normaliza salidas NumPy/CuPy/cuDF/pandas a `np.ndarray`."""
    if hasattr(data, "to_numpy"):
        array = data.to_numpy()
    elif hasattr(data, "get"):
        array = data.get()
    elif hasattr(data, "values_host"):
        array = data.values_host
    else:
        array = np.asarray(data)

    array = np.asarray(array)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def synchronize_gpu() -> None:
    """Sincroniza la GPU para medir tiempos de inferencia de forma consistente."""
    try:  # pragma: no cover - depende del entorno GPU
        import cupy as cp

        cp.cuda.Stream.null.synchronize()
    except Exception:
        return


def maybe_to_sklearn(model: Any) -> Any:
    if hasattr(model, "as_sklearn"):
        return model.as_sklearn()
    return model


def _candidate_iter(
    param_space: dict[str, Iterable[Any]] | list[dict[str, Iterable[Any]]],
    search_type: str,
    n_iter: Optional[int],
    random_state: int,
) -> list[dict[str, Any]]:
    if search_type == "random":
        return list(ParameterSampler(param_space, n_iter=n_iter or 10, random_state=random_state))
    return list(ParameterGrid(param_space))


def _evaluate_candidate(
    estimator_builder: Callable[[dict[str, Any]], Any],
    params: dict[str, Any],
    splits: list[tuple[np.ndarray, np.ndarray]],
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, Any]:
    fold_scores: list[float] = []

    for train_idx, valid_idx in splits:
        model = estimator_builder(params)
        model.fit(X_train[train_idx], y_train[train_idx])
        y_pred = to_numpy(model.predict(X_train[valid_idx]), dtype=y_train.dtype).reshape(-1)
        score = f1_score(y_train[valid_idx], y_pred, average="macro")
        fold_scores.append(float(score))

    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores))
    return {
        "params": params,
        "split_scores": np.asarray(fold_scores, dtype=np.float64),
        "mean_score": mean_score,
        "std_score": std_score,
    }


def manual_search_cv(
    estimator_builder: Callable[[dict[str, Any]], Any],
    param_space: dict[str, Iterable[Any]] | list[dict[str, Iterable[Any]]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv,
    *,
    search_type: str = "grid",
    n_iter: Optional[int] = None,
    random_state: int = 42,
    n_jobs: int = 1,
    verbose: int = 1,
    model_name: str = "Modelo",
) -> SearchResult:
    """
    Reimplementa una búsqueda estilo GridSearchCV/RandomizedSearchCV para
    estimadores GPU de cuML, usando la misma métrica F1 macro.
    """
    candidates = _candidate_iter(param_space, search_type, n_iter, random_state)
    if not candidates:
        raise ValueError("La búsqueda de hiperparámetros no tiene combinaciones.")

    splits = list(cv.split(X_train, y_train))
    if verbose:
        print(f"{model_name}: evaluando {len(candidates)} combinaciones con búsqueda manual...")

    if n_jobs == 1:
        evaluated = [
            _evaluate_candidate(estimator_builder, params, splits, X_train, y_train)
            for params in candidates
        ]
    else:
        evaluated = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_evaluate_candidate)(estimator_builder, params, splits, X_train, y_train)
            for params in candidates
        )

    mean_scores = np.asarray([item["mean_score"] for item in evaluated], dtype=np.float64)
    std_scores = np.asarray([item["std_score"] for item in evaluated], dtype=np.float64)
    params_list = np.asarray([item["params"] for item in evaluated], dtype=object)
    best_index = int(np.argmax(mean_scores))
    best_params = dict(evaluated[best_index]["params"])

    if verbose:
        print(
            f"{model_name}: mejor F1 macro CV = {mean_scores[best_index]:.4f} "
            f"+/- {std_scores[best_index]:.4f}"
        )

    best_model = estimator_builder(best_params)
    best_model.fit(X_train, y_train)

    return SearchResult(
        best_estimator_=best_model,
        best_params_=best_params,
        best_index_=best_index,
        cv_results_={
            "params": params_list,
            "mean_test_score": mean_scores,
            "std_test_score": std_scores,
        },
    )
