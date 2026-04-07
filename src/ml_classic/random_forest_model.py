"""
=============================================================================
 random_forest_model.py — Random Forest optimizado para IoT-IDS en ESP32
=============================================================================
 Compatible con Linux y aceleración GPU opcional mediante RAPIDS/cuML.
 Se conserva `RandomizedSearchCV` en CPU y búsqueda manual en GPU.
=============================================================================
"""

from __future__ import annotations

import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

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


def _cpu_param_distributions() -> dict[str, list]:
    return {
        "n_estimators": [10, 20, 30, 50, 75, 100],
        "max_depth": [5, 8, 10, 12, 15, 20, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", "log2"],
        "criterion": ["gini", "entropy"],
        "class_weight": ["balanced", "balanced_subsample", None],
    }


def _gpu_param_distributions() -> dict[str, list]:
    return {
        "n_estimators": [10, 20, 30, 50, 75, 100],
        "max_depth": [5, 8, 10, 12, 15, 20, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", "log2", None],
        "split_criterion": ["gini", "entropy"],
        "n_bins": [64, 128],
    }


def _build_gpu_random_forest(params):
    from cuml.ensemble import RandomForestClassifier as CuRandomForestClassifier

    return CuRandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        split_criterion=params["split_criterion"],
        n_bins=params["n_bins"],
        random_state=42,
        n_streams=4,
        output_type="numpy",
    )


def train_random_forest(
    *,
    data_dir: str | None = None,
    output_dir: str = ".",
    backend: str = "auto",
    test_size: float = 0.2,
    random_state: int = 42,
    n_jobs: int = -1,
    n_iter: int = 80,
):
    backend_info = resolve_backend(backend)
    print(f"Backend seleccionado para Random Forest: {backend_info.selected} ({backend_info.reason})")

    X_train, X_test, y_train, y_test, scaler = load_dataset(
        scale=False,
        data_dir=data_dir,
        test_size=test_size,
        random_state=random_state,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    effective_jobs = get_default_parallel_jobs(backend_info.selected, n_jobs)

    if backend_info.selected == "gpu":
        print(f"Buscando mejores hiperparámetros para Random Forest con cuML ({n_iter} combinaciones)...")
        search = manual_search_cv(
            _build_gpu_random_forest,
            _gpu_param_distributions(),
            X_train,
            y_train,
            cv,
            search_type="random",
            n_iter=n_iter,
            random_state=random_state,
            n_jobs=effective_jobs,
            verbose=1,
            model_name="Random Forest",
        )
    else:
        print(f"Buscando mejores hiperparámetros para Random Forest con scikit-learn ({n_iter} combinaciones)...")
        search = RandomizedSearchCV(
            RandomForestClassifier(random_state=random_state, n_jobs=effective_jobs),
            _cpu_param_distributions(),
            n_iter=n_iter,
            cv=cv,
            scoring="f1_macro",
            n_jobs=effective_jobs,
            verbose=1,
            refit=True,
            random_state=random_state,
        )
        search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = to_numpy(best_model.predict(X_test), dtype=np.int32)

    best_cv = search.cv_results_["mean_test_score"][search.best_index_]
    std_cv = search.cv_results_["std_test_score"][search.best_index_]
    cv_result = np.array([best_cv - std_cv, best_cv, best_cv + std_cv])

    metrics = print_evaluation(
        "Random Forest",
        y_test,
        y_pred,
        search.best_params_,
        cv_scores=cv_result,
    )

    feature_imp = None
    if hasattr(best_model, "feature_importances_"):
        feature_imp = to_numpy(best_model.feature_importances_, dtype=np.float32)
        sorted_idx = np.argsort(feature_imp)[::-1]
        print("\n Feature Importance (top 5):")
        for i in range(min(5, len(sorted_idx))):
            idx = sorted_idx[i]
            print(f"   {FEATURE_COLUMNS[idx]:>15}: {feature_imp[idx]:.4f}")

    if hasattr(best_model, "estimators_"):
        depths = [tree.get_depth() for tree in best_model.estimators_]
        leaves = [tree.get_n_leaves() for tree in best_model.estimators_]
        total_nodes = sum(tree.tree_.node_count for tree in best_model.estimators_)
        print(f" Árboles: {len(best_model.estimators_)}")
        print(f" Profundidad promedio: {np.mean(depths):.1f} (max {max(depths)})")
        print(f" Hojas promedio:       {np.mean(leaves):.0f} (total {sum(leaves)})")
        est_bytes = total_nodes * 20
        print(f"\n Estimación de memoria ESP32: ~{est_bytes / 1024:.1f} KB ({total_nodes} nodos)")
        if est_bytes > 512 * 1024:
            print(f" ⚠ ADVERTENCIA: Modelo grande para ESP32 (>{est_bytes / 1024:.0f} KB).")
            print("   Considerar reducir n_estimators o max_depth para deployment.")
    else:
        print(
            "\nLa implementación GPU de cuML no expone `estimators_` árbol por árbol; "
            "se omite la estimación detallada de nodos/hojas."
        )

    saved = dump_model_artifacts(
        best_model,
        output_path(output_dir, "random_forest_best.pkl"),
        scaler=scaler,
        scaler_path=output_path(output_dir, "scaler_rf.pkl"),
    )
    saved.extend(save_edge_metadata(output_dir, FEATURE_COLUMNS, CLASS_NAMES, scaler=scaler))
    print("\nArtefactos guardados:")
    for path in saved:
        print(f"  - {path}")

    return best_model, scaler, metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entrena Random Forest para el IDS IoT.")
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
    parser.add_argument("--n-jobs", type=int, default=-1, help="Procesos paralelos para CPU.")
    parser.add_argument(
        "--n-iter",
        type=int,
        default=80,
        help="Número de combinaciones aleatorias a evaluar en la búsqueda.",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train_random_forest(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        backend=args.backend,
        test_size=args.test_size,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        n_iter=args.n_iter,
    )
