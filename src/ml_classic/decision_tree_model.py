"""
=============================================================================
 decision_tree_model.py — Decision Tree optimizado para IoT-IDS en ESP32
=============================================================================
 Ideal para ESP32 por su bajo costo computacional en inferencia (solo
 comparaciones if/else). Se limita la profundidad para caber en memoria.
 class_weight='balanced' compensa el desbalance normal >> bruteforce/scan.
=============================================================================
"""

import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from data_loader import load_dataset, print_evaluation, CLASS_NAMES, FEATURE_COLUMNS


def train_decision_tree():
    X_train, X_test, y_train, y_test, scaler = load_dataset(scale=False)

    param_grid = {
        "max_depth": [5, 8, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "criterion": ["gini", "entropy"],
        "max_features": ["sqrt", "log2", None],
        "class_weight": ["balanced", None],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    print("Buscando mejores hiperparámetros para Decision Tree...")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    cv_scores = grid.cv_results_["mean_test_score"]
    best_cv = cv_scores[grid.best_index_]
    std_cv = grid.cv_results_["std_test_score"][grid.best_index_]
    cv_result = np.array([best_cv - std_cv, best_cv, best_cv + std_cv])

    metrics = print_evaluation(
        "Decision Tree", y_test, y_pred, grid.best_params_,
        cv_scores=cv_result
    )

    depth = best_model.get_depth()
    n_leaves = best_model.get_n_leaves()
    print(f" Profundidad del árbol: {depth}")
    print(f" Número de hojas:       {n_leaves}")
    print(f" Nodos totales:         {best_model.tree_.node_count}")

    feature_imp = best_model.feature_importances_
    sorted_idx = np.argsort(feature_imp)[::-1]
    print("\n Feature Importance (top 5):")
    for i in range(min(5, len(sorted_idx))):
        idx = sorted_idx[i]
        print(f"   {FEATURE_COLUMNS[idx]:>15}: {feature_imp[idx]:.4f}")

    est_bytes = best_model.tree_.node_count * 20
    print(f"\n Estimación memoria ESP32: ~{est_bytes / 1024:.1f} KB")

    joblib.dump(best_model, "decision_tree_best.pkl")
    if scaler:
        joblib.dump(scaler, "scaler_dt.pkl")
    print("\nModelo guardado: decision_tree_best.pkl")

    return best_model, scaler, metrics


if __name__ == "__main__":
    train_decision_tree()
