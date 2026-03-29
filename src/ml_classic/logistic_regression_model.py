"""
=============================================================================
 logistic_regression_model.py — Logistic Regression optimizado para IoT-IDS
=============================================================================
 Muy eficiente en ESP32: la inferencia es solo una multiplicación
 matriz-vector + softmax. Footprint mínimo (~200 bytes para 13x3 pesos).
 Se prueba con regularización L1 (sparse), L2, y ElasticNet.
 class_weight='balanced' compensa el desbalance de clases.
=============================================================================
"""

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from data_loader import load_dataset, print_evaluation, FEATURE_COLUMNS


def train_logistic_regression():
    X_train, X_test, y_train, y_test, scaler = load_dataset(scale=True)

    param_grid = [
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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        LogisticRegression(random_state=42, multi_class="multinomial"),
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    print("Buscando mejores hiperparámetros para Logistic Regression...")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    best_cv = grid.cv_results_["mean_test_score"][grid.best_index_]
    std_cv = grid.cv_results_["std_test_score"][grid.best_index_]
    cv_result = np.array([best_cv - std_cv, best_cv, best_cv + std_cv])

    metrics = print_evaluation(
        "Logistic Regression", y_test, y_pred, grid.best_params_,
        cv_scores=cv_result
    )

    coefs = best_model.coef_
    print(" Coeficientes por clase (top 3 por magnitud):")
    for i, name in enumerate(CLASS_NAMES):
        top_idx = np.argsort(np.abs(coefs[i]))[::-1][:3]
        feats_str = ", ".join(f"{FEATURE_COLUMNS[j]}={coefs[i][j]:.3f}" for j in top_idx)
        print(f"   {name}: {feats_str}")

    n_nonzero = np.count_nonzero(coefs)
    total = coefs.size
    print(f"\n Sparsity: {total - n_nonzero}/{total} coefs son cero "
          f"({(total - n_nonzero)/total:.0%} sparse)")

    mem_bytes = coefs.size * 4 + best_model.intercept_.size * 4 + 13 * 2 * 4
    print(f" Estimación memoria ESP32: ~{mem_bytes} bytes ({mem_bytes/1024:.2f} KB)")

    joblib.dump(best_model, "logistic_regression_best.pkl")
    if scaler:
        joblib.dump(scaler, "scaler_lr.pkl")
    print("\nModelo guardado: logistic_regression_best.pkl")

    return best_model, scaler, metrics


if __name__ == "__main__":
    train_logistic_regression()
