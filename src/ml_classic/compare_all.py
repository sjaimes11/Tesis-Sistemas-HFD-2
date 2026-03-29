"""
=============================================================================
 compare_all.py — Entrena y compara todos los modelos clasicos de ML
=============================================================================
 Ejecuta Decision Tree, Random Forest, Logistic Regression y SVM,
 genera una tabla comparativa y grafica de barras con los resultados.
 
 Ejecutar: python compare_all.py
=============================================================================
"""

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data_loader import load_dataset, print_evaluation, CLASS_NAMES, FEATURE_COLUMNS

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold


def run_all():
    # Cargar datos UNA sola vez, con y sin escalar
    print("Cargando datasets...")
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_dataset(scale=True)
    X_train_raw, X_test_raw, _, _, _ = load_dataset(scale=False)

    # SVM RBF necesita subconjunto para no colgar (~30K por clase max)
    MAX_SVM_RBF = 25000
    n_train = len(X_train_scaled)
    if n_train > MAX_SVM_RBF * 3:
        rng = np.random.RandomState(42)
        svm_idx = rng.choice(n_train, size=MAX_SVM_RBF * 3, replace=False)
        X_train_svm_rbf = X_train_scaled[svm_idx]
        y_train_svm_rbf = y_train[svm_idx]
        print(f"\nSVM RBF: submuestreado a {len(svm_idx)} para evitar O(n^3)\n")
    else:
        X_train_svm_rbf = X_train_scaled
        y_train_svm_rbf = y_train

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Decision Tree": {
            "search": "grid",
            "estimator": DecisionTreeClassifier(random_state=42),
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
            "search": "random",
            "n_iter": 60,
            "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
            "params": {
                "n_estimators": [10, 30, 50, 100],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "max_features": ["sqrt", "log2"],
                "class_weight": ["balanced", "balanced_subsample", None],
            },
            "X_train": X_train_raw,
            "X_test": X_test_raw,
            "y_train": y_train,
        },
        "Logistic Regression": {
            "search": "grid",
            "estimator": LogisticRegression(
                random_state=42, max_iter=2000, multi_class="multinomial"
            ),
            "params": {
                "C": [0.01, 0.1, 1, 10, 100],
                "penalty": ["l2"],
                "solver": ["lbfgs"],
                "class_weight": ["balanced", None],
            },
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
        },
        "SVM (Linear)": {
            "search": "grid",
            "estimator": LinearSVC(random_state=42, dual="auto", max_iter=5000),
            "params": {
                "C": [0.01, 0.1, 1, 10],
                "class_weight": ["balanced", None],
            },
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
        },
        "SVM (RBF)": {
            "search": "grid",
            "estimator": SVC(kernel="rbf", random_state=42),
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
        print(f"\n{'#'*60}")
        print(f" Entrenando: {name}")
        print(f"{'#'*60}")

        t0 = time.time()

        if cfg.get("search") == "random":
            searcher = RandomizedSearchCV(
                cfg["estimator"], cfg["params"],
                n_iter=cfg.get("n_iter", 50),
                cv=cv, scoring="f1_macro", n_jobs=-1,
                verbose=0, refit=True, random_state=42,
            )
        else:
            searcher = GridSearchCV(
                cfg["estimator"], cfg["params"],
                cv=cv, scoring="f1_macro", n_jobs=-1,
                verbose=0, refit=True,
            )

        searcher.fit(cfg["X_train"], cfg["y_train"])
        train_time = time.time() - t0

        y_pred = searcher.best_estimator_.predict(cfg["X_test"])

        # Inferencia precisa con perf_counter y 100 iteraciones
        sample = cfg["X_test"][:1]
        _ = searcher.best_estimator_.predict(sample)  # warmup
        t_inf_start = time.perf_counter()
        for _ in range(100):
            searcher.best_estimator_.predict(sample)
        inf_time_us = (time.perf_counter() - t_inf_start) / 100 * 1e6

        best_cv = searcher.cv_results_["mean_test_score"][searcher.best_index_]
        std_cv = searcher.cv_results_["std_test_score"][searcher.best_index_]

        metrics = print_evaluation(
            name, y_test, y_pred, searcher.best_params_,
            cv_scores=np.array([best_cv - std_cv, best_cv, best_cv + std_cv])
        )
        metrics["train_time_s"] = train_time
        metrics["inference_us"] = inf_time_us
        metrics["best_params"] = searcher.best_params_
        metrics["cv_mean"] = best_cv
        metrics["cv_std"] = std_cv

        results[name] = metrics

    # --- Tabla Comparativa ---
    print(f"\n{'='*90}")
    print(f" TABLA COMPARATIVA DE MODELOS")
    print(f"{'='*90}")
    header = (f"  {'Modelo':<22} {'Accuracy':>9} {'F1 macro':>9} {'F1 wgt':>9} "
              f"{'Prec':>9} {'Recall':>9} {'MCC':>9} {'CV±std':>12}")
    print(header)
    print(f"  {'-'*88}")
    for name, m in results.items():
        print(f"  {name:<22} {m['accuracy']:>9.4f} {m['f1_macro']:>9.4f} {m['f1_weighted']:>9.4f} "
              f"{m['precision']:>9.4f} {m['recall']:>9.4f} {m['mcc']:>9.4f} "
              f"{m['cv_mean']:>.4f}±{m['cv_std']:.4f}")

    best_name = max(results, key=lambda k: results[k]["f1_macro"])
    print(f"\n  Mejor modelo por F1 macro: {best_name} ({results[best_name]['f1_macro']:.4f})")

    # --- Exportar a CSV ---
    rows = []
    for name, m in results.items():
        rows.append({
            "modelo": name,
            "accuracy": m["accuracy"],
            "f1_macro": m["f1_macro"],
            "f1_weighted": m["f1_weighted"],
            "precision": m["precision"],
            "recall": m["recall"],
            "mcc": m["mcc"],
            "cv_mean": m["cv_mean"],
            "cv_std": m["cv_std"],
            "train_time_s": m["train_time_s"],
            "inference_us": m["inference_us"],
        })
    df_results = pd.DataFrame(rows)
    df_results.to_csv("comparison_results.csv", index=False)
    print("Resultados guardados: comparison_results.csv")

    # --- Graficas ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    names = list(results.keys())
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    x = np.arange(len(names))
    w = 0.25

    # Grafica 1: Accuracy vs F1
    axes[0].bar(x - w/2, [results[n]["accuracy"] for n in names], w,
                label="Accuracy", color="#2196F3", alpha=0.85)
    axes[0].bar(x + w/2, [results[n]["f1_macro"] for n in names], w,
                label="F1 macro", color="#FF9800", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Accuracy vs F1 Macro")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(axis="y", alpha=0.3)

    # Grafica 2: Precision / Recall / MCC
    axes[1].bar(x - w, [results[n]["precision"] for n in names], w,
                label="Precision", color="#4CAF50", alpha=0.85)
    axes[1].bar(x, [results[n]["recall"] for n in names], w,
                label="Recall", color="#E91E63", alpha=0.85)
    axes[1].bar(x + w, [results[n]["mcc"] for n in names], w,
                label="MCC", color="#607D8B", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("Score")
    axes[1].set_title("Precision / Recall / MCC")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(axis="y", alpha=0.3)

    # Grafica 3: Tiempo de inferencia
    inf_times = [results[n]["inference_us"] for n in names]
    bars = axes[2].barh(names, inf_times, color=colors[:len(names)], alpha=0.85)
    for bar, val in zip(bars, inf_times):
        axes[2].text(bar.get_width() + max(inf_times)*0.02, bar.get_y() + bar.get_height()/2,
                     f"{val:.0f} us", va="center", fontsize=8)
    axes[2].set_xlabel("Tiempo de inferencia (us)")
    axes[2].set_title("Latencia de Inferencia (1 muestra)")
    axes[2].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig("comparison_results.png", dpi=150, bbox_inches="tight")
    print("Grafica guardada: comparison_results.png\n")

    return results


if __name__ == "__main__":
    run_all()
