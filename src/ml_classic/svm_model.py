"""
=============================================================================
 svm_model.py — Support Vector Machine optimizado para IoT-IDS en ESP32
=============================================================================
 Compatible con Linux y aceleración GPU opcional usando RAPIDS/cuML para
 LinearSVC y SVC(RBF).
=============================================================================
"""

from __future__ import annotations

import argparse
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC, SVC

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


def _linear_params() -> dict[str, list]:
    return {
        "C": [0.01, 0.1, 1, 10, 100],
        "loss": ["hinge", "squared_hinge"],
        "class_weight": ["balanced", None],
        "max_iter": [5000],
    }


def _rbf_params() -> dict[str, list]:
    return {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.01, 0.1],
        "class_weight": ["balanced", None],
    }


def _build_gpu_linear_svm(params):
    from cuml.svm import LinearSVC as CuLinearSVC

    return CuLinearSVC(
        C=params["C"],
        loss=params["loss"],
        class_weight=params.get("class_weight"),
        max_iter=params["max_iter"],
        penalty="l2",
        multi_class="ovr",
        output_type="numpy",
    )


def _build_gpu_rbf_svm(params):
    from cuml.svm import SVC as CuSVC

    return CuSVC(
        kernel="rbf",
        C=params["C"],
        gamma=params["gamma"],
        class_weight=params.get("class_weight"),
        output_type="numpy",
    )


def _coef_and_intercept(model):
    coef = getattr(model, "coef_", None)
    intercept = getattr(model, "intercept_", None)
    if coef is None and hasattr(model, "coef"):
        coef = model.coef
    if intercept is None and hasattr(model, "intercept"):
        intercept = model.intercept
    return coef, intercept


def train_svm(
    *,
    data_dir: str | None = None,
    output_dir: str = ".",
    backend: str = "auto",
    test_size: float = 0.2,
    random_state: int = 42,
    n_jobs: int = -1,
    max_samples_per_class: int = 30000,
):
    backend_info = resolve_backend(backend)
    print(f"Backend seleccionado para SVM: {backend_info.selected} ({backend_info.reason})")

    X_train, X_test, y_train, y_test, scaler = load_dataset(
        scale=True,
        max_samples_per_class=max_samples_per_class,
        data_dir=data_dir,
        test_size=test_size,
        random_state=random_state,
    )

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    effective_jobs = get_default_parallel_jobs(backend_info.selected, n_jobs)

    print("\n" + "=" * 60)
    print(" Fase 1: Linear SVM (óptimo para ESP32)")
    print("=" * 60)

    if backend_info.selected == "gpu":
        print("Buscando mejores hiperparámetros para Linear SVM con cuML...")
        linear_search = manual_search_cv(
            _build_gpu_linear_svm,
            _linear_params(),
            X_train,
            y_train,
            cv,
            search_type="grid",
            random_state=random_state,
            n_jobs=effective_jobs,
            verbose=1,
            model_name="SVM Linear",
        )
    else:
        print("Buscando mejores hiperparámetros para Linear SVM con scikit-learn...")
        linear_search = GridSearchCV(
            LinearSVC(random_state=random_state, dual="auto"),
            _linear_params(),
            cv=cv,
            scoring="f1_macro",
            n_jobs=effective_jobs,
            verbose=1,
            refit=True,
        )
        linear_search.fit(X_train, y_train)

    best_linear = linear_search.best_estimator_
    y_pred_linear = to_numpy(best_linear.predict(X_test), dtype=np.int32)
    best_cv_l = linear_search.cv_results_["mean_test_score"][linear_search.best_index_]
    std_cv_l = linear_search.cv_results_["std_test_score"][linear_search.best_index_]

    metrics_linear = print_evaluation(
        "SVM (Linear)",
        y_test,
        y_pred_linear,
        linear_search.best_params_,
        cv_scores=np.array([best_cv_l - std_cv_l, best_cv_l, best_cv_l + std_cv_l]),
    )

    coef, intercept = _coef_and_intercept(best_linear)
    mem_linear = 0
    if coef is not None and intercept is not None:
        coef_np = to_numpy(coef, dtype=np.float32)
        intercept_np = to_numpy(intercept, dtype=np.float32)
        mem_linear = coef_np.size * 4 + intercept_np.size * 4
        print(f" Memoria ESP32 (Linear): ~{mem_linear} bytes")
    results["linear"] = (best_linear, metrics_linear, mem_linear)

    print("\n" + "=" * 60)
    print(" Fase 2: RBF SVM (mayor capacidad, más costoso en ESP32)")
    print("=" * 60)

    if backend_info.selected == "gpu":
        print("Buscando mejores hiperparámetros para RBF SVM con cuML...")
        rbf_search = manual_search_cv(
            _build_gpu_rbf_svm,
            _rbf_params(),
            X_train,
            y_train,
            cv,
            search_type="grid",
            random_state=random_state,
            n_jobs=effective_jobs,
            verbose=1,
            model_name="SVM RBF",
        )
    else:
        print("Buscando mejores hiperparámetros para RBF SVM con scikit-learn...")
        rbf_search = GridSearchCV(
            SVC(kernel="rbf", random_state=random_state),
            _rbf_params(),
            cv=cv,
            scoring="f1_macro",
            n_jobs=effective_jobs,
            verbose=1,
            refit=True,
        )
        rbf_search.fit(X_train, y_train)

    best_rbf = rbf_search.best_estimator_
    y_pred_rbf = to_numpy(best_rbf.predict(X_test), dtype=np.int32)
    best_cv_r = rbf_search.cv_results_["mean_test_score"][rbf_search.best_index_]
    std_cv_r = rbf_search.cv_results_["std_test_score"][rbf_search.best_index_]

    metrics_rbf = print_evaluation(
        "SVM (RBF)",
        y_test,
        y_pred_rbf,
        rbf_search.best_params_,
        cv_scores=np.array([best_cv_r - std_cv_r, best_cv_r, best_cv_r + std_cv_r]),
    )

    mem_rbf = 0
    n_sv = None
    support_vectors = getattr(best_rbf, "support_vectors_", None)
    n_support = getattr(best_rbf, "n_support_", None)
    if support_vectors is not None:
        support_vectors_np = to_numpy(support_vectors, dtype=np.float32)
        n_sv = support_vectors_np.shape[0]
        mem_rbf = n_sv * support_vectors_np.shape[1] * 4 + n_sv * 4 + 3 * 4
        print(f" Support Vectors: {n_sv}")
        print(f" Memoria ESP32 (RBF): ~{mem_rbf / 1024:.1f} KB")
        if mem_rbf > 512 * 1024:
            print(" WARNING: Demasiados SVs para ESP32. Considerar Linear SVM.")
    elif n_support is not None:
        n_support_np = to_numpy(n_support, dtype=np.int32).reshape(-1)
        n_sv = int(np.sum(n_support_np))
        print(f" Support Vectors: {n_sv} ({n_support_np.tolist()})")
    results["rbf"] = (best_rbf, metrics_rbf, mem_rbf)

    print("\n" + "=" * 60)
    print(" Comparación Linear vs RBF")
    print("=" * 60)
    print(f"  {'Métrica':<15} {'Linear':>10} {'RBF':>10}")
    print(f"  {'-' * 35}")
    for key in ["accuracy", "f1_macro", "f1_weighted", "precision", "recall", "mcc"]:
        print(f"  {key:<15} {metrics_linear[key]:>10.4f} {metrics_rbf[key]:>10.4f}")
    print(f"  {'memoria':<15} {mem_linear:>8} B {mem_rbf:>8} B")

    if metrics_rbf["f1_macro"] - metrics_linear["f1_macro"] < 0.02:
        print("\n -> Recomendación: Linear SVM (rendimiento similar, mucho menor footprint)")
        best_model = best_linear
        best_name = "linear"
    else:
        print("\n -> Recomendación: RBF SVM (mejora significativa justifica el costo)")
        best_model = best_rbf
        best_name = "rbf"

    saved_paths = []
    saved_paths.extend(
        dump_model_artifacts(best_model, output_path(output_dir, "svm_best.pkl"))
    )
    saved_paths.extend(
        dump_model_artifacts(best_linear, output_path(output_dir, "svm_linear_best.pkl"))
    )
    saved_paths.extend(
        dump_model_artifacts(best_rbf, output_path(output_dir, "svm_rbf_best.pkl"))
    )
    saved_paths.extend(
        dump_model_artifacts(
            best_model,
            output_path(output_dir, "svm_selected.pkl"),
            scaler=scaler,
            scaler_path=output_path(output_dir, "scaler_svm.pkl"),
        )
    )

    unique_paths = []
    seen = set()
    for path in saved_paths:
        if path not in seen:
            unique_paths.append(path)
            seen.add(path)

    for path in save_edge_metadata(output_dir, FEATURE_COLUMNS, CLASS_NAMES, scaler=scaler):
        if path not in seen:
            unique_paths.append(path)
            seen.add(path)

    print(f"\nModelos guardados (seleccionado: {best_name}):")
    for path in unique_paths:
        print(f"  - {path}")

    return best_model, scaler, results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entrena SVM lineal y RBF para el IDS IoT.")
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
        "--max-samples-per-class",
        type=int,
        default=30000,
        help="Límite por clase para mantener el costo del SVM bajo control.",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train_svm(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        backend=args.backend,
        test_size=args.test_size,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        max_samples_per_class=args.max_samples_per_class,
    )
