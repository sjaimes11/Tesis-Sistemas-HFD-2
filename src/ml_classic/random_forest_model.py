"""
=============================================================================
 random_forest_model.py — Random Forest optimizado para IoT-IDS en ESP32
=============================================================================
 Se limita n_estimators y max_depth para que el modelo exportado a C
 quepa en la flash del ESP32 (~4MB). Con 10-20 árboles poco profundos
 se logra buen accuracy sin explotar la memoria.
 
 Usa RandomizedSearchCV en lugar de GridSearchCV para evitar ~3600 fits.
=============================================================================
"""

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from data_loader import load_dataset, print_evaluation, FEATURE_COLUMNS


def train_random_forest():
    X_train, X_test, y_train, y_test, scaler = load_dataset(scale=False)

    param_distributions = {
        "n_estimators": [10, 20, 30, 50, 75, 100],
        "max_depth": [5, 8, 10, 12, 15, 20, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", "log2"],
        "criterion": ["gini", "entropy"],
        "class_weight": ["balanced", "balanced_subsample", None],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_distributions,
        n_iter=80,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        refit=True,
        random_state=42,
    )

    print("Buscando mejores hiperparámetros para Random Forest (80 combinaciones)...")
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    best_cv = search.cv_results_["mean_test_score"][search.best_index_]
    std_cv = search.cv_results_["std_test_score"][search.best_index_]
    cv_result = np.array([best_cv - std_cv, best_cv, best_cv + std_cv])

    metrics = print_evaluation(
        "Random Forest", y_test, y_pred, search.best_params_,
        cv_scores=cv_result
    )

    depths = [t.get_depth() for t in best_model.estimators_]
    leaves = [t.get_n_leaves() for t in best_model.estimators_]
    print(f" Árboles: {len(best_model.estimators_)}")
    print(f" Profundidad promedio: {np.mean(depths):.1f} (max {max(depths)})")
    print(f" Hojas promedio:       {np.mean(leaves):.0f} (total {sum(leaves)})")

    feature_imp = best_model.feature_importances_
    sorted_idx = np.argsort(feature_imp)[::-1]
    print("\n Feature Importance (top 5):")
    for i in range(min(5, len(sorted_idx))):
        idx = sorted_idx[i]
        print(f"   {FEATURE_COLUMNS[idx]:>15}: {feature_imp[idx]:.4f}")

    total_nodes = sum(t.tree_.node_count for t in best_model.estimators_)
    est_bytes = total_nodes * 20
    print(f"\n Estimación de memoria ESP32: ~{est_bytes / 1024:.1f} KB ({total_nodes} nodos)")
    if est_bytes > 512 * 1024:
        print(f" ⚠ ADVERTENCIA: Modelo grande para ESP32 (>{est_bytes/1024:.0f} KB).")
        print(f"   Considerar reducir n_estimators o max_depth para deployment.")

    joblib.dump(best_model, "random_forest_best.pkl")
    if scaler:
        joblib.dump(scaler, "scaler_rf.pkl")
    print("\nModelo guardado: random_forest_best.pkl")

    return best_model, scaler, metrics


if __name__ == "__main__":
    train_random_forest()
