"""
=============================================================================
 svm_model.py — Support Vector Machine optimizado para IoT-IDS en ESP32
=============================================================================
 Se prueban kernels linear y RBF. Para ESP32, LinearSVC es más práctico
 (inferencia = dot product), mientras que RBF requiere guardar support
 vectors (puede ser grande). Se evalúan ambos y se reporta el trade-off.
 
 Limita muestras para RBF (O(n^2-n^3) en entrenamiento).
=============================================================================
"""

import joblib
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from data_loader import load_dataset, print_evaluation, FEATURE_COLUMNS


def train_svm():
    X_train, X_test, y_train, y_test, scaler = load_dataset(
        scale=True,
        max_samples_per_class=30000
    )

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --- Linear SVM ---
    print("\n" + "="*60)
    print(" Fase 1: Linear SVM (optimo para ESP32)")
    print("="*60)

    param_grid_linear = {
        "C": [0.01, 0.1, 1, 10, 100],
        "loss": ["hinge", "squared_hinge"],
        "class_weight": ["balanced", None],
        "max_iter": [5000],
    }

    grid_linear = GridSearchCV(
        LinearSVC(random_state=42, dual="auto"),
        param_grid_linear,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    print("Buscando mejores hiperparámetros para Linear SVM...")
    grid_linear.fit(X_train, y_train)

    best_linear = grid_linear.best_estimator_
    y_pred_linear = best_linear.predict(X_test)

    best_cv_l = grid_linear.cv_results_["mean_test_score"][grid_linear.best_index_]
    std_cv_l = grid_linear.cv_results_["std_test_score"][grid_linear.best_index_]

    metrics_linear = print_evaluation(
        "SVM (Linear)", y_test, y_pred_linear, grid_linear.best_params_,
        cv_scores=np.array([best_cv_l - std_cv_l, best_cv_l, best_cv_l + std_cv_l])
    )

    mem_linear = best_linear.coef_.size * 4 + best_linear.intercept_.size * 4
    print(f" Memoria ESP32 (Linear): ~{mem_linear} bytes")
    results["linear"] = (best_linear, metrics_linear, mem_linear)

    # --- RBF SVM ---
    print("\n" + "="*60)
    print(" Fase 2: RBF SVM (mayor capacidad, mas costoso en ESP32)")
    print("="*60)

    param_grid_rbf = {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.01, 0.1],
        "class_weight": ["balanced", None],
    }

    grid_rbf = GridSearchCV(
        SVC(kernel="rbf", random_state=42),
        param_grid_rbf,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    print("Buscando mejores hiperparámetros para RBF SVM...")
    grid_rbf.fit(X_train, y_train)

    best_rbf = grid_rbf.best_estimator_
    y_pred_rbf = best_rbf.predict(X_test)

    best_cv_r = grid_rbf.cv_results_["mean_test_score"][grid_rbf.best_index_]
    std_cv_r = grid_rbf.cv_results_["std_test_score"][grid_rbf.best_index_]

    metrics_rbf = print_evaluation(
        "SVM (RBF)", y_test, y_pred_rbf, grid_rbf.best_params_,
        cv_scores=np.array([best_cv_r - std_cv_r, best_cv_r, best_cv_r + std_cv_r])
    )

    n_sv = best_rbf.n_support_.sum()
    mem_rbf = n_sv * 13 * 4 + n_sv * 4 + 3 * 4
    print(f" Support Vectors: {n_sv} ({best_rbf.n_support_})")
    print(f" Memoria ESP32 (RBF): ~{mem_rbf / 1024:.1f} KB")
    if mem_rbf > 512 * 1024:
        print(f" WARNING: Demasiados SVs para ESP32. Considerar Linear SVM.")
    results["rbf"] = (best_rbf, metrics_rbf, mem_rbf)

    # --- Seleccion final ---
    print("\n" + "="*60)
    print(" Comparacion Linear vs RBF")
    print("="*60)
    print(f"  {'Metrica':<15} {'Linear':>10} {'RBF':>10}")
    print(f"  {'-'*35}")
    for key in ["accuracy", "f1_macro", "f1_weighted", "precision", "recall", "mcc"]:
        print(f"  {key:<15} {metrics_linear[key]:>10.4f} {metrics_rbf[key]:>10.4f}")
    print(f"  {'memoria':<15} {mem_linear:>8} B {mem_rbf:>8} B")

    if metrics_rbf["f1_macro"] - metrics_linear["f1_macro"] < 0.02:
        print("\n -> Recomendacion: Linear SVM (rendimiento similar, mucho menor footprint)")
        best_model = best_linear
        best_name = "linear"
    else:
        print("\n -> Recomendacion: RBF SVM (mejora significativa justifica el costo)")
        best_model = best_rbf
        best_name = "rbf"

    joblib.dump(best_model, "svm_best.pkl")
    joblib.dump(best_linear, "svm_linear_best.pkl")
    joblib.dump(best_rbf, "svm_rbf_best.pkl")
    if scaler:
        joblib.dump(scaler, "scaler_svm.pkl")
    print(f"\nModelos guardados: svm_best.pkl (seleccionado: {best_name})")

    return best_model, scaler, results


if __name__ == "__main__":
    train_svm()
