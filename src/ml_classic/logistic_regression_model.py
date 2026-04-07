"""
=============================================================================
 logistic_regression_model.py — Logistic Regression optimizado para IoT-IDS
=============================================================================
 Compatible con Linux y aceleración GPU opcional usando RAPIDS/cuML.
 Uso típico en servidor:

   python logistic_regression_model.py --backend auto --data-dir ~/Tesis_Sistemas/Data_Sets
=============================================================================
"""

from __future__ import annotations

import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

try:
    from .data_loader import CLASS_NAMES, FEATURE_COLUMNS, load_dataset, print_evaluation
    from .training_runtime import (
        dump_model_artifacts,
        get_default_parallel_jobs,
        manual_search_cv,
        output_path,
        resolve_backend,
        save_edge_metadata,
        to_numpy,
    )
except ImportError:
    from data_loader import CLASS_NAMES, FEATURE_COLUMNS, load_dataset, print_evaluation
    from training_runtime import (
        dump_model_artifacts,
        get_default_parallel_jobs,
        manual_search_cv,
        output_path,
        resolve_backend,
        save_edge_metadata,
        to_numpy,
    )


def _cpu_param_grid() -> list[dict[str, list]]:
    return [
        {
            "penalty": ["l2"],
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "solver": ["lbfgs", "newton-cg"],
            "class_weight": ["balanced", None],
            "max_iter": [1000],
        },
        {
            "penalty": ["l1"],
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "solver": ["saga"],
            "class_weight": ["balanced", None],
            "max_iter": [2000],
        },
        {
            "penalty": ["elasticnet"],
            "C": [0.01, 0.1, 1, 10],
            "solver": ["saga"],
            "l1_ratio": [0.25, 0.5, 0.75],
            "class_weight": ["balanced", None],
            "max_iter": [2000],
        },
    ]


def _gpu_param_grid() -> list[dict[str, list]]:
    return [
        {
            "penalty": ["l2"],
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "class_weight": ["balanced", None],
            "max_iter": [1000, 2000],
        },
        {
            "penalty": ["l1"],
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "class_weight": ["balanced", None],
            "max_iter": [1000, 2000],
        },
        {
            "penalty": ["elasticnet"],
            "C": [0.01, 0.1, 1, 10],
            "l1_ratio": [0.25, 0.5, 0.75],
            "class_weight": ["balanced", None],
            "max_iter": [1000, 2000],
        },
    ]


def _build_gpu_logistic(params):
    from cuml.linear_model import LogisticRegression as CuLogisticRegression

    return CuLogisticRegression(
        penalty=params["penalty"],
        C=params["C"],
        class_weight=params.get("class_weight"),
        max_iter=params["max_iter"],
        l1_ratio=params.get("l1_ratio"),
        solver="qn",
        output_type="numpy",
    )


def train_logistic_regression(
    *,
    data_dir: str | None = None,
    output_dir: str = ".",
    backend: str = "auto",
    test_size: float = 0.2,
    random_state: int = 42,
    n_jobs: int = -1,
):
    backend_info = resolve_backend(backend)
    print(f"Backend seleccionado para Logistic Regression: {backend_info.selected} ({backend_info.reason})")

    X_train, X_test, y_train, y_test, scaler = load_dataset(
        scale=True,
        data_dir=data_dir,
        test_size=test_size,
        random_state=random_state,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    effective_jobs = get_default_parallel_jobs(backend_info.selected, n_jobs)

    if backend_info.selected == "gpu":
        print("Buscando mejores hiperparámetros para Logistic Regression con cuML...")
        search = manual_search_cv(
            _build_gpu_logistic,
            _gpu_param_grid(),
            X_train,
            y_train,
            cv,
            search_type="grid",
            random_state=random_state,
            n_jobs=effective_jobs,
            verbose=1,
            model_name="Logistic Regression",
        )
    else:
        print("Buscando mejores hiperparámetros para Logistic Regression con scikit-learn...")
        search = GridSearchCV(
            LogisticRegression(random_state=random_state, multi_class="multinomial"),
            _cpu_param_grid(),
            cv=cv,
            scoring="f1_macro",
            n_jobs=effective_jobs,
            verbose=1,
            refit=True,
        )
        search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = to_numpy(best_model.predict(X_test), dtype=np.int32)

    best_cv = search.cv_results_["mean_test_score"][search.best_index_]
    std_cv = search.cv_results_["std_test_score"][search.best_index_]
    cv_result = np.array([best_cv - std_cv, best_cv, best_cv + std_cv])

    metrics = print_evaluation(
        "Logistic Regression",
        y_test,
        y_pred,
        search.best_params_,
        cv_scores=cv_result,
    )

    coefs = to_numpy(best_model.coef_, dtype=np.float32)
    intercept = to_numpy(best_model.intercept_, dtype=np.float32)

    print(" Coeficientes por clase (top 3 por magnitud):")
    for i, name in enumerate(CLASS_NAMES):
        top_idx = np.argsort(np.abs(coefs[i]))[::-1][:3]
        feats_str = ", ".join(f"{FEATURE_COLUMNS[j]}={coefs[i][j]:.3f}" for j in top_idx)
        print(f"   {name}: {feats_str}")

    n_nonzero = np.count_nonzero(coefs)
    total = coefs.size
    print(
        f"\n Sparsity: {total - n_nonzero}/{total} coefs son cero "
        f"({(total - n_nonzero) / total:.0%} sparse)"
    )

    mem_bytes = coefs.size * 4 + intercept.size * 4 + len(FEATURE_COLUMNS) * 2 * 4
    print(f" Estimación memoria ESP32: ~{mem_bytes} bytes ({mem_bytes / 1024:.2f} KB)")

    saved = dump_model_artifacts(
        best_model,
        output_path(output_dir, "logistic_regression_best.pkl"),
        scaler=scaler,
        scaler_path=output_path(output_dir, "scaler_lr.pkl"),
    )
    saved.extend(save_edge_metadata(output_dir, FEATURE_COLUMNS, CLASS_NAMES, scaler=scaler))
    print("\nArtefactos guardados:")
    for path in saved:
        print(f"  - {path}")

    return best_model, scaler, metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entrena Logistic Regression para el IDS IoT.")
    parser.add_argument("--data-dir", default=None, help="Directorio con los CSVs de entrenamiento.")
    parser.add_argument("--output-dir", default=".", help="Directorio donde se guardan los modelos.")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Backend de entrenamiento: auto detecta cuML y usa GPU si está disponible.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporción reservada para test.")
    parser.add_argument("--random-state", type=int, default=42, help="Semilla de reproducibilidad.")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Procesos paralelos para CPU. En GPU se fuerza a 1 para no sobrecargar la VRAM.",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train_logistic_regression(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        backend=args.backend,
        test_size=args.test_size,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
