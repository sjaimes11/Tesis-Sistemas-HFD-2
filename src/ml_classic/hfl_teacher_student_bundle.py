"""
=============================================================================
 hfl_teacher_student_bundle.py — Bundles HFL v7 para modelos clasicos
=============================================================================
 Entrena un modelo clasico "teacher" y destila su comportamiento en una MLP
 compatible con la arquitectura HFL v7 existente:

   13 -> 32 -> 16 -> 8 -> 3

 El artefacto desplegable queda en formato Keras (`ids_3class.keras`) para
 que el gateway pueda seguir reentrenando por rondas, y tambien se exporta a
 `model_weights.h` para ESP32 con el mismo estilo de `hfl_v7-RN`.

 Teachers soportados:
 - random_forest
 - knn
 - naive_bayes
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

try:
    from .data_loader import CLASS_NAMES, FEATURE_COLUMNS, load_dataset
except ImportError:
    from data_loader import CLASS_NAMES, FEATURE_COLUMNS, load_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
RN_TEMPLATE_DIR = REPO_ROOT / "hfl_v7-RN"


@dataclass(frozen=True)
class ModelSpec:
    kind: str
    display_name: str
    bundle_dir_name: str
    teacher_filename: str
    max_samples_per_class: int | None
    distill_samples: int | None
    teacher_eval_samples: int | None


MODEL_SPECS: dict[str, ModelSpec] = {
    "random_forest": ModelSpec(
        kind="random_forest",
        display_name="Random Forest",
        bundle_dir_name="hfl_v7-random-forest",
        teacher_filename="teacher_random_forest.joblib",
        max_samples_per_class=None,
        distill_samples=None,
        teacher_eval_samples=None,
    ),
    "knn": ModelSpec(
        kind="knn",
        display_name="K-Nearest Neighbors",
        bundle_dir_name="hfl_v7-knn",
        teacher_filename="teacher_knn.joblib",
        max_samples_per_class=30000,
        distill_samples=15000,
        teacher_eval_samples=12000,
    ),
    "naive_bayes": ModelSpec(
        kind="naive_bayes",
        display_name="Gaussian Naive Bayes",
        bundle_dir_name="hfl_v7-naive-bayes",
        teacher_filename="teacher_naive_bayes.joblib",
        max_samples_per_class=None,
        distill_samples=None,
        teacher_eval_samples=None,
    ),
}


def require_tensorflow():
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "No se pudo importar TensorFlow. En este proyecto suele pasar cuando "
            "TensorFlow fue instalado con NumPy 1.x y el entorno actual tiene "
            "NumPy 2.x. Corrige el venv con: "
            "`pip install 'numpy<2' 'tensorflow==2.19.0'` "
            "o recrea el entorno antes de entrenar."
        ) from exc
    return tf


def set_global_seed(seed: int) -> None:
    tf = require_tensorflow()
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def ensure_clean_bundle_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    shutil.copytree(
        RN_TEMPLATE_DIR,
        path,
        ignore=shutil.ignore_patterns("__pycache__", "Results", "ids_3class.keras", "model_weights.h"),
    )
    (path / "Results").mkdir(exist_ok=True)


def patch_gateway_loaders(bundle_dir: Path) -> None:
    replacements = {
        'print("[GATEWAY] Cargando modelo base ids_3class.keras...")':
        'print("[GATEWAY] Cargando modelo base ids_3class.keras / ids_3class.h5...")',
        'print(f"[FOG-{FOG_ROLE.upper()}] Cargando modelo base ids_3class.keras...")':
        'print(f"[FOG-{FOG_ROLE.upper()}] Cargando modelo base ids_3class.keras / ids_3class.h5...")',
        'model = tf.keras.models.load_model("ids_3class.keras")':
        'model = tf.keras.models.load_model("ids_3class.keras" if __import__("os").path.exists("ids_3class.keras") else "ids_3class.h5")',
    }
    for script_name in ["gateway_hfl.py", "gateway_hfl_fog.py"]:
        script_path = bundle_dir / script_name
        text = script_path.read_text(encoding="utf-8")
        for old, new in replacements.items():
            text = text.replace(old, new)
        script_path.write_text(text, encoding="utf-8")


def load_scaled_dataset(
    *,
    data_dir: str | None,
    max_samples_per_class: int | None,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    X_train, X_test, y_train, y_test, _ = load_dataset(
        max_samples_per_class=max_samples_per_class,
        test_size=test_size,
        random_state=random_state,
        scale=False,
        data_dir=data_dir,
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def build_teacher(kind: str, random_state: int) -> Any:
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=160,
            max_depth=14,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=random_state,
        )
    if kind == "knn":
        return KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            algorithm="auto",
            n_jobs=1,
        )
    if kind == "naive_bayes":
        return GaussianNB(var_smoothing=1e-9)
    raise ValueError(f"Modelo no soportado: {kind}")


def build_student_model(input_dim: int, learning_rate: float) -> tf.keras.Model:
    tf = require_tensorflow()
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,), name="input_features"),
            tf.keras.layers.Dense(32, activation="relu", name="dense_0"),
            tf.keras.layers.Dense(16, activation="relu", name="dense_1"),
            tf.keras.layers.Dense(8, activation="relu", name="dense_3"),
            tf.keras.layers.Dense(3, activation="softmax", name="dense_out"),
        ],
        name="ids_student_mlp",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def sample_balanced_indices(y: np.ndarray, max_total: int, random_state: int) -> np.ndarray:
    if max_total is None or max_total >= len(y):
        return np.arange(len(y))

    rng = np.random.default_rng(random_state)
    per_class = max(1, max_total // len(CLASS_NAMES))
    picks: list[np.ndarray] = []
    for label in range(len(CLASS_NAMES)):
        indices = np.where(y == label)[0]
        take = min(len(indices), per_class)
        picks.append(rng.choice(indices, size=take, replace=False))

    picked = np.concatenate(picks)
    rng.shuffle(picked)
    return picked


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def export_student_to_header(model: tf.keras.Model, scaler: StandardScaler, output_path: Path, teacher_name: str) -> None:
    dense_0 = model.get_layer("dense_0")
    dense_1 = model.get_layer("dense_1")
    dense_3 = model.get_layer("dense_3")
    dense_out = model.get_layer("dense_out")

    w1, b1 = dense_0.get_weights()
    w2, b2 = dense_1.get_weights()
    w3, b3 = dense_3.get_weights()
    w4, b4 = dense_out.get_weights()

    def format_1d(name: str, values: np.ndarray) -> str:
        joined = ", ".join(f"{float(value):.6f}f" for value in values.tolist())
        return f"static const float {name}[{len(values)}] = {{{joined}}};"

    def format_2d(name: str, values: np.ndarray) -> str:
        rows = []
        for row in values.tolist():
            rows.append("  {" + ", ".join(f"{float(value):.6f}f" for value in row) + "}")
        return (
            f"static const float {name}[{values.shape[0]}][{values.shape[1]}] = {{\n"
            + ",\n".join(rows)
            + "\n};"
        )

    header = [
        "// AUTO-GENERADO - NO EDITAR MANUALMENTE",
        f"// Teacher original: {teacher_name}",
        "// Student deployable: MLP 13->32->16->8->3 compatible con HFL v7",
        "// Clases: normal, mqtt_bruteforce, scan_A",
        "#pragma once",
        "",
        "#include <stddef.h>",
        "",
        "static const size_t FEATURE_COUNT = 13;",
        "static const size_t NUM_CLASSES = 3;",
        "",
        format_1d("scaler_mean", scaler.mean_.astype(np.float32)),
        format_1d("scaler_std", scaler.scale_.astype(np.float32)),
        "",
        "// Dense 1: (13, 32)",
        format_2d("W1_base", w1.astype(np.float32)),
        format_1d("b1_base", b1.astype(np.float32)),
        "",
        "// Dense 2: (32, 16)",
        format_2d("W2_base", w2.astype(np.float32)),
        format_1d("b2_base", b2.astype(np.float32)),
        "",
        "// Dense 3: (16, 8)",
        format_2d("W3_base", w3.astype(np.float32)),
        format_1d("b3_base", b3.astype(np.float32)),
        "",
        "// Dense out: (8, 3)",
        format_2d("W4_base", w4.astype(np.float32)),
        format_1d("b4_base", b4.astype(np.float32)),
        "",
    ]
    output_path.write_text("\n".join(header), encoding="utf-8")


def write_feature_metadata(bundle_dir: Path, scaler: StandardScaler) -> None:
    feature_order = bundle_dir / "feature_order.csv"
    feature_order.write_text("feature\n" + "\n".join(FEATURE_COLUMNS) + "\n", encoding="utf-8")

    label_map = {str(index): name for index, name in enumerate(CLASS_NAMES)}
    (bundle_dir / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    scaler_payload = {
        "mean": [float(value) for value in scaler.mean_],
        "std": [float(value) for value in scaler.scale_],
        "features": FEATURE_COLUMNS,
    }
    (bundle_dir / "scaler_params.json").write_text(json.dumps(scaler_payload, indent=2), encoding="utf-8")


def write_lineage_docs(bundle_dir: Path, spec: ModelSpec, summary: dict[str, Any]) -> None:
    architecture = f"""# HFL v7 - {spec.display_name}

Esta carpeta mantiene el estilo de `hfl_v7-RN`, pero el modelo base no es una RN entrenada directamente sobre etiquetas duras.

## Esquema teacher -> student

- Teacher original: `{spec.display_name}`
- Student desplegable: MLP `13 -> 32 -> 16 -> 8 -> 3`
- Formato del student:
  - Linux recomendado: `ids_3class.keras`
  - Fallback estable en este entorno Windows: `ids_3class.h5`
- Export TinyML: `model_weights.h`

## Por que asi

La carpeta `hfl_v7-RN` reentrena el modelo por rondas intercambiando pesos de capas densas (`W3`, `b3`, `W4`, `b4`).
Los modelos clasicos como Random Forest, KNN y Naive Bayes no encajan nativamente en ese protocolo.
Por eso aqui se usa un teacher clasico para entrenar una red estudiante con la misma arquitectura que ya usa tu gateway.

## Resultado mas reciente

- Teacher accuracy: {summary["teacher_metrics"]["accuracy"]:.4f}
- Student accuracy: {summary["student_metrics"]["accuracy"]:.4f}
- Teacher F1 macro: {summary["teacher_metrics"]["f1_macro"]:.4f}
- Student F1 macro: {summary["student_metrics"]["f1_macro"]:.4f}

## Artefactos

- `ids_3class.h5`: student model compatible con el gateway HFL.
- `ids_3class.keras`: opcional, si se activa guardado nativo de Keras desde Linux.
- `model_weights.h`: pesos base para ESP32.
- `{spec.teacher_filename}`: teacher clasico original.
- `training_summary.json`: metricas y configuracion del entrenamiento.
"""
    (bundle_dir / "ARCHITECTURE.md").write_text(architecture, encoding="utf-8")


def create_notebook(bundle_dir: Path, spec: ModelSpec) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# HFL v7 - {spec.display_name}\n",
                    "\n",
                    "Este notebook entrena un teacher clasico y lo destila a una MLP compatible con `hfl_v7-RN`.\n",
                    "El resultado se guarda en esta misma carpeta como `ids_3class.keras` y `model_weights.h`.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "import os\n",
                    "import sys\n",
                    "\n",
                    "def find_repo_root(start: Path) -> Path:\n",
                    "    explicit = os.environ.get('HFL_REPO_ROOT') or os.environ.get('ML_CLASSIC_REPO_ROOT')\n",
                    "    if explicit:\n",
                    "        candidate = Path(explicit).expanduser().resolve()\n",
                    "        if candidate.exists():\n",
                    "            return candidate\n",
                    "\n",
                    "    markers = ('ml_classic', 'ml_classic_v2')\n",
                    "    search_roots = [start, *start.parents]\n",
                    "    for candidate in search_roots:\n",
                    "        if any((candidate / marker).exists() for marker in markers):\n",
                    "            return candidate\n",
                    "\n",
                    "    for candidate in search_roots:\n",
                    "        for marker in markers:\n",
                    "            marker_dir = candidate / marker\n",
                    "            if (marker_dir / 'hfl_teacher_student_bundle.py').exists():\n",
                    "                return candidate\n",
                    "\n",
                    "    raise RuntimeError('No se pudo localizar la raiz del proyecto.')\n",
                    "\n",
                    "REPO_ROOT = find_repo_root(Path.cwd().resolve())\n",
                    "if str(REPO_ROOT) not in sys.path:\n",
                    "    sys.path.insert(0, str(REPO_ROOT))\n",
                    "\n",
                    "if (REPO_ROOT / 'ml_classic').exists():\n",
                    "    from ml_classic.hfl_teacher_student_bundle import train_teacher_student_bundle\n",
                    "elif (REPO_ROOT / 'ml_classic_v2').exists():\n",
                    "    from ml_classic_v2.hfl_teacher_student_bundle import train_teacher_student_bundle\n",
                    "else:\n",
                    "    raise RuntimeError('No se encontro ni ml_classic ni ml_classic_v2 dentro de la raiz detectada.')\n",
                    "\n",
                    "if (REPO_ROOT / 'Data').exists():\n",
                    "    DATA_DIR = str(REPO_ROOT / 'Data')\n",
                    "elif (REPO_ROOT / 'Data_Sets').exists():\n",
                    "    DATA_DIR = str(REPO_ROOT / 'Data_Sets')\n",
                    "else:\n",
                    "    DATA_DIR = '/home/sajaimesp/Tesis_Sistemas/Data_Sets'\n",
                    f"BUNDLE_DIR = str(REPO_ROOT / '{spec.bundle_dir_name}')\n",
                    f"MODEL_KIND = '{spec.kind}'\n",
                    "RANDOM_STATE = 42\n",
                    "TEST_SIZE = 0.2\n",
                    f"MAX_SAMPLES_PER_CLASS = {repr(spec.max_samples_per_class)}\n",
                    "SAVE_NATIVE_KERAS = False  # En Linux puedes ponerlo en True.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "summary = train_teacher_student_bundle(\n",
                    "    model_kind=MODEL_KIND,\n",
                    "    data_dir=DATA_DIR,\n",
                    "    bundle_dir=BUNDLE_DIR,\n",
                    "    random_state=RANDOM_STATE,\n",
                    "    test_size=TEST_SIZE,\n",
                    "    max_samples_per_class=MAX_SAMPLES_PER_CLASS,\n",
                    "    save_native_keras=SAVE_NATIVE_KERAS,\n",
                    ")\n",
                    "summary\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path = bundle_dir / f"train_{spec.kind}.ipynb"
    notebook_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def train_teacher_student_bundle(
    *,
    model_kind: str,
    data_dir: str | None = None,
    bundle_dir: str | None = None,
    random_state: int = 42,
    test_size: float = 0.2,
    max_samples_per_class: int | None = None,
    learning_rate: float = 0.003,
    soft_epochs: int = 16,
    hard_epochs: int = 8,
    save_native_keras: bool = False,
) -> dict[str, Any]:
    tf = require_tensorflow()
    if model_kind not in MODEL_SPECS:
        raise ValueError(f"Modelo no soportado: {model_kind}")

    spec = MODEL_SPECS[model_kind]
    chosen_max_samples = max_samples_per_class if max_samples_per_class is not None else spec.max_samples_per_class
    final_bundle_dir = Path(bundle_dir).resolve() if bundle_dir else (REPO_ROOT / spec.bundle_dir_name)

    set_global_seed(random_state)
    ensure_clean_bundle_dir(final_bundle_dir)
    patch_gateway_loaders(final_bundle_dir)

    X_train, X_test, y_train, y_test, scaler = load_scaled_dataset(
        data_dir=data_dir,
        max_samples_per_class=chosen_max_samples,
        test_size=test_size,
        random_state=random_state,
    )

    teacher = build_teacher(model_kind, random_state=random_state)
    print(f"\n[TEACHER] Entrenando {spec.display_name}...")
    teacher.fit(X_train, y_train)

    teacher_eval_indices = sample_balanced_indices(y_test, spec.teacher_eval_samples, random_state)
    teacher_pred = teacher.predict(X_test[teacher_eval_indices])
    teacher_metrics = evaluate_predictions(y_test[teacher_eval_indices], teacher_pred)
    print(f"[TEACHER] Accuracy={teacher_metrics['accuracy']:.4f} | F1-macro={teacher_metrics['f1_macro']:.4f}")

    distill_indices = sample_balanced_indices(y_train, spec.distill_samples, random_state)
    X_distill = X_train[distill_indices]
    teacher_soft = teacher.predict_proba(X_distill).astype(np.float32)

    student = build_student_model(X_train.shape[1], learning_rate=learning_rate)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        )
    ]

    print(f"[STUDENT] Distillation soft-labels sobre {len(X_distill)} muestras...")
    student.fit(
        X_distill,
        teacher_soft,
        epochs=soft_epochs,
        batch_size=256,
        validation_split=0.1,
        verbose=0,
        callbacks=callbacks,
    )

    print(f"[STUDENT] Fine-tuning con etiquetas reales sobre {len(X_train)} muestras...")
    student.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate * 0.75),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    student.fit(
        X_train,
        y_train,
        epochs=hard_epochs,
        batch_size=256,
        validation_split=0.1,
        verbose=0,
        callbacks=callbacks,
    )

    student_pred = np.argmax(student.predict(X_test, verbose=0), axis=1)
    student_metrics = evaluate_predictions(y_test, student_pred)
    print(f"[STUDENT] Accuracy={student_metrics['accuracy']:.4f} | F1-macro={student_metrics['f1_macro']:.4f}")

    keras_path = final_bundle_dir / "ids_3class.keras"
    h5_path = final_bundle_dir / "ids_3class.h5"
    header_path = final_bundle_dir / "model_weights.h"
    teacher_path = final_bundle_dir / spec.teacher_filename

    print(f"[SAVE] Guardando fallback H5 en {h5_path}...")
    student.save(h5_path, include_optimizer=False)
    print("[SAVE] ids_3class.h5 listo.")
    if save_native_keras:
        print(f"[SAVE] Intentando guardar Keras nativo en {keras_path}...")
        student.save(keras_path, include_optimizer=False)
        print("[SAVE] ids_3class.keras listo.")
    joblib.dump(teacher, teacher_path)
    print(f"[SAVE] Teacher guardado en {teacher_path}.")
    export_student_to_header(student, scaler, header_path, teacher_name=spec.display_name)
    print(f"[SAVE] Header ESP32 listo en {header_path}.")
    write_feature_metadata(final_bundle_dir, scaler)
    print("[SAVE] Metadatos guardados.")

    summary = {
        "model_kind": model_kind,
        "display_name": spec.display_name,
        "bundle_dir": str(final_bundle_dir),
        "teacher_path": str(teacher_path),
        "h5_path": str(h5_path),
        "keras_path": str(keras_path) if save_native_keras and keras_path.exists() else None,
        "header_path": str(header_path),
        "teacher_metrics": teacher_metrics,
        "student_metrics": student_metrics,
        "config": {
            "random_state": random_state,
            "test_size": test_size,
            "max_samples_per_class": chosen_max_samples,
            "learning_rate": learning_rate,
            "soft_epochs": soft_epochs,
            "hard_epochs": hard_epochs,
            "save_native_keras": save_native_keras,
            "distill_samples": spec.distill_samples,
            "teacher_eval_samples": spec.teacher_eval_samples,
        },
        "retraining_note": {
            "random_forest": "El Random Forest exacto no soporta partial_fit en sklearn. En este bundle se reentrena por rondas la MLP estudiante.",
            "knn": "KNN exacto puede crecer agregando muestras, pero no se agrega elegantemente en FL. En este bundle se reentrena por rondas la MLP estudiante.",
            "naive_bayes": "Gaussian Naive Bayes si soporta actualizacion incremental, pero aqui se mantiene la MLP estudiante para seguir el protocolo HFL v7.",
        }[model_kind],
    }
    (final_bundle_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_lineage_docs(final_bundle_dir, spec, summary)
    create_notebook(final_bundle_dir, spec)
    tf.keras.backend.clear_session()
    print("[DONE] Bundle generado.")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entrena teachers clasicos y genera bundles HFL v7 compatibles con Keras/ESP32."
    )
    parser.add_argument(
        "model_kind",
        choices=[*MODEL_SPECS.keys(), "all"],
        help="Teacher clasico a entrenar.",
    )
    parser.add_argument("--data-dir", default=None, help="Directorio con los CSV de entrenamiento.")
    parser.add_argument("--bundle-root", default=str(REPO_ROOT), help="Raiz donde se crearan los bundles.")
    parser.add_argument("--random-state", type=int, default=42, help="Semilla global.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporcion de test.")
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Limite opcional por clase. Si no se define, usa el valor recomendado por modelo.",
    )
    parser.add_argument("--soft-epochs", type=int, default=16, help="Epocas de distillation con soft labels.")
    parser.add_argument("--hard-epochs", type=int, default=8, help="Epocas de fine-tuning con etiquetas reales.")
    parser.add_argument(
        "--save-native-keras",
        action="store_true",
        help="Intenta guardar tambien ids_3class.keras. En este Windows puede bloquearse; en Linux suele funcionar bien.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    bundle_root = Path(args.bundle_root).resolve()
    kinds = list(MODEL_SPECS.keys()) if args.model_kind == "all" else [args.model_kind]

    for kind in kinds:
        spec = MODEL_SPECS[kind]
        train_teacher_student_bundle(
            model_kind=kind,
            data_dir=args.data_dir,
            bundle_dir=str(bundle_root / spec.bundle_dir_name),
            random_state=args.random_state,
            test_size=args.test_size,
            max_samples_per_class=args.max_samples_per_class,
            soft_epochs=args.soft_epochs,
            hard_epochs=args.hard_epochs,
            save_native_keras=args.save_native_keras,
        )


if __name__ == "__main__":
    main()
