"""
=============================================================================
 decision_tree_model.py — Decision Tree optimizado para IoT-IDS en ESP32
=============================================================================
 Se mantiene en CPU porque cuML no expone un DecisionTreeClassifier
 standalone equivalente para entrenamiento general. Aun así, el script ya es
 portable a Linux y permite configurar rutas/salidas por CLI.
=============================================================================
"""

from __future__ import annotations

import argparse
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

try:
    from .data_loader import CLASS_NAMES, FEATURE_COLUMNS, load_dataset, print_evaluation
    from .training_runtime import (
        dump_model_artifacts,
        output_path,
        resolve_backend,
        save_edge_metadata,
        to_numpy,
    )
except ImportError:
    from data_loader import CLASS_NAMES, FEATURE_COLUMNS, load_dataset, print_evaluation
    from training_runtime import (
        dump_model_artifacts,
        output_path,
        resolve_backend,
        save_edge_metadata,
        to_numpy,
    )


def train_decision_tree(
    *,
    data_dir: str | None = None,
    output_dir: str = ".",
    backend: str = "auto",
    test_size: float = 0.2,
    random_state: int = 42,
    n_jobs: int = -1,
):
    backend_info = resolve_backend(backend)
    if backend_info.selected == "gpu":
        print(
            "Decision Tree se entrenará en CPU. "
            "La aceleración GPU queda reservada para Logistic Regression, SVM y Random Forest."
        )
    else:
        print(f"Backend seleccionado para Decision Tree: {backend_info.selected} ({backend_info.reason})")

    X_train, X_test, y_train, y_test, scaler = load_dataset(
        scale=False,
        data_dir=data_dir,
        test_size=test_size,
        random_state=random_state,
    )

    param_grid = {
        "max_depth": [5, 8, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "criterion": ["gini", "entropy"],
        "max_features": ["sqrt", "log2", None],
        "class_weight": ["balanced", None],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=random_state),
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=n_jobs,
        verbose=1,
        refit=True,
    )

    print("Buscando mejores hiperparámetros para Decision Tree...")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = to_numpy(best_model.predict(X_test), dtype=np.int32)

    best_cv = grid.cv_results_["mean_test_score"][grid.best_index_]
    std_cv = grid.cv_results_["std_test_score"][grid.best_index_]
    cv_result = np.array([best_cv - std_cv, best_cv, best_cv + std_cv])

    metrics = print_evaluation(
        "Decision Tree",
        y_test,
        y_pred,
        grid.best_params_,
        cv_scores=cv_result,
    )

    depth = best_model.get_depth()
    n_leaves = best_model.get_n_leaves()
    print(f" Profundidad del árbol: {depth}")
    print(f" Número de hojas:       {n_leaves}")
    print(f" Nodos totales:         {best_model.tree_.node_count}")

    feature_imp = to_numpy(best_model.feature_importances_, dtype=np.float32)
    sorted_idx = np.argsort(feature_imp)[::-1]
    print("\n Feature Importance (top 5):")
    for i in range(min(5, len(sorted_idx))):
        idx = sorted_idx[i]
        print(f"   {FEATURE_COLUMNS[idx]:>15}: {feature_imp[idx]:.4f}")

    est_bytes = best_model.tree_.node_count * 20
    print(f"\n Estimación memoria ESP32: ~{est_bytes / 1024:.1f} KB")

    saved = dump_model_artifacts(
        best_model,
        output_path(output_dir, "decision_tree_best.pkl"),
        scaler=scaler,
        scaler_path=output_path(output_dir, "scaler_dt.pkl"),
    )
    saved.extend(save_edge_metadata(output_dir, FEATURE_COLUMNS, CLASS_NAMES, scaler=scaler))
    print("\nArtefactos guardados:")
    for path in saved:
        print(f"  - {path}")

    return best_model, scaler, metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entrena un Decision Tree para el IDS IoT.")
    parser.add_argument("--data-dir", default=None, help="Directorio con los CSVs de entrenamiento.")
    parser.add_argument("--output-dir", default=".", help="Directorio donde se guardan los modelos.")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Se acepta por consistencia, pero Decision Tree se mantiene en CPU.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporción reservada para test.")
    parser.add_argument("--random-state", type=int, default=42, help="Semilla de reproducibilidad.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Procesos paralelos para GridSearchCV.")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train_decision_tree(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        backend=args.backend,
        test_size=args.test_size,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
