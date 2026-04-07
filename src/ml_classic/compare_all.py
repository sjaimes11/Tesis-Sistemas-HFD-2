"""
=============================================================================
 compare_all.py — Entrena y compara todos los modelos clásicos de ML
=============================================================================
 Portable a Linux y con backend GPU opcional para Logistic Regression,
 Random Forest y SVM cuando RAPIDS/cuML está disponible.
=============================================================================
"""

from __future__ import annotations

import argparse
import time

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .data_loader import load_dataset, print_evaluation
    from .training_runtime import (
        ensure_output_dir,
        get_default_parallel_jobs,
        manual_search_cv,
        output_path,
        resolve_backend,
        synchronize_gpu,
        to_numpy,
    )
except ImportError:
    from data_loader import load_dataset, print_evaluation
    from training_runtime import (
        ensure_output_dir,
        get_default_parallel_jobs,
        manual_search_cv,
        output_path,
        resolve_backend,
        synchronize_gpu,
        to_numpy,
    )


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


def _search_model(cfg, cv, random_state, cpu_jobs):
    search_type = cfg.get("search", "grid")
    backend = cfg.get("backend", "cpu")

    if backend == "gpu":
        return manual_search_cv(
            cfg["builder"],
            cfg["params"],
            cfg["X_train"],
            cfg["y_train"],
            cv,
            search_type=search_type,
            n_iter=cfg.get("n_iter"),
            random_state=random_state,
            n_jobs=1,
            verbose=1,
            model_name=cfg["name"],
        )

    if search_type == "random":
        searcher = RandomizedSearchCV(
            cfg["estimator"],
            cfg["params"],
            n_iter=cfg.get("n_iter", 50),
            cv=cv,
            scoring="f1_macro",
            n_jobs=cpu_jobs,
            verbose=0,
            refit=True,
            random_state=random_state,
        )
    else:
        searcher = GridSearchCV(
            cfg["estimator"],
            cfg["params"],
            cv=cv,
            scoring="f1_macro",
            n_jobs=cpu_jobs,
            verbose=0,
            refit=True,
        )

    searcher.fit(cfg["X_train"], cfg["y_train"])
    return searcher


def run_all(
    *,
    data_dir: str | None = None,
    output_dir: str = ".",
    backend: str = "auto",
    test_size: float = 0.2,
    random_state: int = 42,
    n_jobs: int = -1,
    svm_rbf_max_samples_per_class: int = 25000,
):
    backend_info = resolve_backend(backend)
    cpu_jobs = get_default_parallel_jobs("cpu", n_jobs)
    output_dir = ensure_output_dir(output_dir)

    print(f"Backend global solicitado: {backend}")
    print(f"Backend global resuelto: {backend_info.selected} ({backend_info.reason})")

    print("Cargando datasets...")
    X_train_scaled, X_test_scaled, y_train, y_test, _ = load_dataset(
        scale=True,
        data_dir=data_dir,
        test_size=test_size,
        random_state=random_state,
    )
    X_train_raw, X_test_raw, _, _, _ = load_dataset(
        scale=False,
        data_dir=data_dir,
        test_size=test_size,
        random_state=random_state,
    )

    max_svm_samples = svm_rbf_max_samples_per_class * 3
    n_train = len(X_train_scaled)
    if n_train > max_svm_samples:
        rng = np.random.RandomState(random_state)
        svm_idx = rng.choice(n_train, size=max_svm_samples, replace=False)
        X_train_svm_rbf = X_train_scaled[svm_idx]
        y_train_svm_rbf = y_train[svm_idx]
        print(f"\nSVM RBF: submuestreado a {len(svm_idx)} filas para evitar costos O(n^3)\n")
    else:
        X_train_svm_rbf = X_train_scaled
        y_train_svm_rbf = y_train

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    gpu_enabled = backend_info.selected == "gpu"

    models = {
        "Decision Tree": {
            "name": "Decision Tree",
            "backend": "cpu",
            "search": "grid",
            "estimator": DecisionTreeClassifier(random_state=random_state),
            "params": {
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 5],
                "criterion": ["gini", "entropy"],
                "class_weight": ["balanced", None],
            },
            "X_train": X_train_raw,
            "X_test": X_test_raw,
            "y_train": y_train,
        },
        "Random Forest": {
            "name": "Random Forest",
            "backend": "gpu" if gpu_enabled else "cpu",
            "search": "random",
            "n_iter": 60,
            "estimator": RandomForestClassifier(random_state=random_state, n_jobs=cpu_jobs),
            "builder": _build_gpu_random_forest,
            "params": (
                {
                    "n_estimators": [10, 30, 50, 100],
                    "max_depth": [5, 10, 15, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 5],
                    "max_features": ["sqrt", "log2"],
                    "class_weight": ["balanced", "balanced_subsample", None],
                }
                if not gpu_enabled
                else {
                    "n_estimators": [10, 30, 50, 100],
                    "max_depth": [5, 10, 15, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 5],
                    "max_features": ["sqrt", "log2", None],
                    "split_criterion": ["gini", "entropy"],
                    "n_bins": [64, 128],
                }
            ),
            "X_train": X_train_raw,
            "X_test": X_test_raw,
            "y_train": y_train,
        },
        "Logistic Regression": {
            "name": "Logistic Regression",
            "backend": "gpu" if gpu_enabled else "cpu",
            "search": "grid",
            "estimator": LogisticRegression(
                random_state=random_state,
                max_iter=2000,
                multi_class="multinomial",
            ),
            "builder": _build_gpu_logistic,
            "params": (
                {
                    "C": [0.01, 0.1, 1, 10, 100],
                    "penalty": ["l2"],
                    "solver": ["lbfgs"],
                    "class_weight": ["balanced", None],
                    "max_iter": [2000],
                }
                if not gpu_enabled
                else {
                    "penalty": ["l2", "l1"],
                    "C": [0.01, 0.1, 1, 10, 100],
                    "class_weight": ["balanced", None],
                    "max_iter": [1000, 2000],
                }
            ),
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
        },
        "SVM (Linear)": {
            "name": "SVM (Linear)",
            "backend": "gpu" if gpu_enabled else "cpu",
            "search": "grid",
            "estimator": LinearSVC(random_state=random_state, dual="auto", max_iter=5000),
            "builder": _build_gpu_linear_svm,
            "params": {
                "C": [0.01, 0.1, 1, 10],
                "loss": ["hinge", "squared_hinge"],
                "class_weight": ["balanced", None],
                "max_iter": [5000],
            },
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
        },
        "SVM (RBF)": {
            "name": "SVM (RBF)",
            "backend": "gpu" if gpu_enabled else "cpu",
            "search": "grid",
            "estimator": SVC(kernel="rbf", random_state=random_state),
            "builder": _build_gpu_rbf_svm,
            "params": {
                "C": [0.1, 1, 10],
                "gamma": ["scale", "auto"],
                "class_weight": ["balanced", None],
            },
            "X_train": X_train_svm_rbf,
            "X_test": X_test_scaled,
            "y_train": y_train_svm_rbf,
        },
    }

    results = {}

    for name, cfg in models.items():
        print(f"\n{'#' * 60}")
        print(f" Entrenando: {name} [{cfg['backend']}]")
        print(f"{'#' * 60}")

        t0 = time.time()
        searcher = _search_model(cfg, cv, random_state, cpu_jobs)
        train_time = time.time() - t0

        y_pred = to_numpy(searcher.best_estimator_.predict(cfg["X_test"]), dtype=np.int32)

        sample = cfg["X_test"][:1]
        _ = searcher.best_estimator_.predict(sample)
        synchronize_gpu()
        t_inf_start = time.perf_counter()
        for _ in range(100):
            searcher.best_estimator_.predict(sample)
        synchronize_gpu()
        inf_time_us = (time.perf_counter() - t_inf_start) / 100 * 1e6

        best_cv = searcher.cv_results_["mean_test_score"][searcher.best_index_]
        std_cv = searcher.cv_results_["std_test_score"][searcher.best_index_]

        metrics = print_evaluation(
            name,
            y_test,
            y_pred,
            searcher.best_params_,
            cv_scores=np.array([best_cv - std_cv, best_cv, best_cv + std_cv]),
        )
        metrics["train_time_s"] = train_time
        metrics["inference_us"] = inf_time_us
        metrics["best_params"] = searcher.best_params_
        metrics["cv_mean"] = best_cv
        metrics["cv_std"] = std_cv
        metrics["backend"] = cfg["backend"]

        results[name] = metrics

    print(f"\n{'=' * 110}")
    print(" TABLA COMPARATIVA DE MODELOS")
    print(f"{'=' * 110}")
    header = (
        f"  {'Modelo':<22} {'Backend':<8} {'Accuracy':>9} {'F1 macro':>9} "
        f"{'F1 wgt':>9} {'Prec':>9} {'Recall':>9} {'MCC':>9} {'CV±std':>12}"
    )
    print(header)
    print(f"  {'-' * 108}")
    for name, metrics in results.items():
        print(
            f"  {name:<22} {metrics['backend']:<8} {metrics['accuracy']:>9.4f} "
            f"{metrics['f1_macro']:>9.4f} {metrics['f1_weighted']:>9.4f} "
            f"{metrics['precision']:>9.4f} {metrics['recall']:>9.4f} "
            f"{metrics['mcc']:>9.4f} {metrics['cv_mean']:.4f}±{metrics['cv_std']:.4f}"
        )

    best_name = max(results, key=lambda key: results[key]["f1_macro"])
    print(f"\n  Mejor modelo por F1 macro: {best_name} ({results[best_name]['f1_macro']:.4f})")

    rows = []
    for name, metrics in results.items():
        rows.append(
            {
                "modelo": name,
                "backend": metrics["backend"],
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "f1_weighted": metrics["f1_weighted"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "mcc": metrics["mcc"],
                "cv_mean": metrics["cv_mean"],
                "cv_std": metrics["cv_std"],
                "train_time_s": metrics["train_time_s"],
                "inference_us": metrics["inference_us"],
            }
        )

    df_results = pd.DataFrame(rows)
    csv_path = output_path(output_dir, "comparison_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"Resultados guardados: {csv_path}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    names = list(results.keys())
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    x = np.arange(len(names))
    w = 0.25

    axes[0].bar(
        x - w / 2,
        [results[name]["accuracy"] for name in names],
        w,
        label="Accuracy",
        color="#2196F3",
        alpha=0.85,
    )
    axes[0].bar(
        x + w / 2,
        [results[name]["f1_macro"] for name in names],
        w,
        label="F1 macro",
        color="#FF9800",
        alpha=0.85,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Accuracy vs F1 Macro")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(
        x - w,
        [results[name]["precision"] for name in names],
        w,
        label="Precision",
        color="#4CAF50",
        alpha=0.85,
    )
    axes[1].bar(
        x,
        [results[name]["recall"] for name in names],
        w,
        label="Recall",
        color="#E91E63",
        alpha=0.85,
    )
    axes[1].bar(
        x + w,
        [results[name]["mcc"] for name in names],
        w,
        label="MCC",
        color="#607D8B",
        alpha=0.85,
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("Score")
    axes[1].set_title("Precision / Recall / MCC")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(axis="y", alpha=0.3)

    inf_times = [results[name]["inference_us"] for name in names]
    bars = axes[2].barh(names, inf_times, color=colors[: len(names)], alpha=0.85)
    for bar, val in zip(bars, inf_times):
        axes[2].text(
            bar.get_width() + max(inf_times) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.0f} us",
            va="center",
            fontsize=8,
        )
    axes[2].set_xlabel("Tiempo de inferencia (us)")
    axes[2].set_title("Latencia de Inferencia (1 muestra)")
    axes[2].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plot_path = output_path(output_dir, "comparison_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Gráfica guardada: {plot_path}\n")

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entrena y compara todos los modelos clásicos.")
    parser.add_argument("--data-dir", default=None, help="Directorio con los CSVs de entrenamiento.")
    parser.add_argument("--output-dir", default=".", help="Directorio donde se guardan CSV y gráficas.")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Backend global: auto detecta cuML y usa GPU si está disponible.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporción reservada para test.")
    parser.add_argument("--random-state", type=int, default=42, help="Semilla de reproducibilidad.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Procesos paralelos para CPU.")
    parser.add_argument(
        "--svm-rbf-max-samples-per-class",
        type=int,
        default=25000,
        help="Límite por clase para el RBF SVM dentro del benchmark comparativo.",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_all(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        backend=args.backend,
        test_size=args.test_size,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        svm_rbf_max_samples_per_class=args.svm_rbf_max_samples_per_class,
    )
