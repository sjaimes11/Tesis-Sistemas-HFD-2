# -*- coding: utf-8 -*-
import os
import json
import random
import warnings
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from keras import layers, callbacks

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)

# ============================================================
# 1. RUTAS
# ============================================================
# Si subiste los CSV manualmente a Colab, deja esto así:
DATA_DIR = Path("/content").resolve()

# Carpeta de salida separada
OUTPUT_DIR = Path("/content/cnn1d_outputs").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

required_files = [
    "uniflow_normal.csv",
    "uniflow_mqtt_bruteforce.csv",
    "uniflow_scan_A.csv",
]

missing = [name for name in required_files if not (DATA_DIR / name).exists()]
if missing:
    raise FileNotFoundError(f"No encontré estos archivos en {DATA_DIR}: {missing}")

print("DATA_DIR   =", DATA_DIR)
print("OUTPUT_DIR =", OUTPUT_DIR)

# ============================================================
# 2. CARGAR DATOS
# ============================================================
print("\nCargando datos...")
df_normal = pd.read_csv(DATA_DIR / "uniflow_normal.csv", low_memory=False)
df_bruteforce = pd.read_csv(DATA_DIR / "uniflow_mqtt_bruteforce.csv", low_memory=False)
df_scan = pd.read_csv(DATA_DIR / "uniflow_scan_A.csv", low_memory=False)

df_normal["class_label"] = 0
df_bruteforce["class_label"] = 1
df_scan["class_label"] = 2

print(f"  Normal:          {len(df_normal)}")
print(f"  MQTT Bruteforce: {len(df_bruteforce)}")
print(f"  Scan A:          {len(df_scan)}")

# ============================================================
# 3. FEATURES
# ============================================================
FLOW_FEATURES = [
    "num_pkts", "mean_iat", "std_iat", "min_iat", "max_iat",
    "mean_pkt_len", "num_bytes", "num_psh_flags", "num_rst_flags",
    "num_urg_flags", "std_pkt_len", "min_pkt_len", "max_pkt_len",
]
TARGET = "class_label"
CLASS_NAMES = ["normal", "mqtt_bruteforce", "scan_A"]
NUM_CLASSES = 3
FEATURE_COUNT = len(FLOW_FEATURES)

# ============================================================
# 4. BALANCEAR CLASES
# ============================================================
min_count = min(len(df_normal), len(df_bruteforce), len(df_scan))
print(f"\nBalanceando clases a {min_count} muestras cada una...")

df_normal_s = df_normal.sample(n=min_count, random_state=SEED)
df_bruteforce_s = df_bruteforce.sample(n=min_count, random_state=SEED)
df_scan_s = df_scan.sample(n=min_count, random_state=SEED)

df = pd.concat([df_normal_s, df_bruteforce_s, df_scan_s], ignore_index=True)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)

print(f"Dataset final: {df.shape}")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name}: {(df[TARGET] == i).sum()}")

# ============================================================
# 5. SPLIT Y ESCALAR
# ============================================================
X = df[FLOW_FEATURES].values.astype(np.float32)
y = df[TARGET].values.astype(np.int32)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y,
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train).astype(np.float32)
X_test_sc = scaler.transform(X_test).astype(np.float32)

print(f"\nTrain: {X_train_sc.shape} | Test: {X_test_sc.shape}")

# ============================================================
# 6. MODELO MLP
# ============================================================
print(f"\nArquitectura MLP: {FEATURE_COUNT} -> 32 -> BN -> 16 -> BN -> 8 -> {NUM_CLASSES}")

model = keras.Sequential([
    layers.Input(shape=(FEATURE_COUNT,), name="input_features"),
    layers.Dense(32, activation="relu", name="dense_1"),
    layers.BatchNormalization(name="bn_1"),
    layers.Dropout(0.3, name="drop_1"),
    layers.Dense(16, activation="relu", name="dense_2"),
    layers.BatchNormalization(name="bn_2"),
    layers.Dropout(0.2, name="drop_2"),
    layers.Dense(8, activation="relu", name="dense_3"),
    layers.Dense(NUM_CLASSES, activation="softmax", name="dense_out"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=False,
)

model.summary()

# ============================================================
# 7. ENTRENAMIENTO
# ============================================================
early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
)

checkpoint_path = OUTPUT_DIR / "ids_3class_best.keras"
model_ckpt = callbacks.ModelCheckpoint(
    filepath=str(checkpoint_path),
    monitor="val_loss",
    save_best_only=True,
)

history = model.fit(
    X_train_sc,
    y_train,
    validation_split=0.15,
    epochs=100,
    batch_size=256,
    callbacks=[early_stop, reduce_lr, model_ckpt],
    verbose=1,
)

# ============================================================
# 8. EVALUACIÓN
# ============================================================
y_pred = model.predict(X_test_sc, verbose=0).argmax(axis=1)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print(f"\nAccuracy: {acc:.4f}")
print(f"F1 (weighted): {f1:.4f}\n")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
)
plt.title("Confusion Matrix - MLP 3-Class IDS")
plt.ylabel("Real")
plt.xlabel("Predicho")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history["accuracy"], label="Train")
axes[0].plot(history.history["val_accuracy"], label="Val")
axes[0].set_title("Accuracy MLP")
axes[0].set_xlabel("Epocas")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history["loss"], label="Train")
axes[1].plot(history.history["val_loss"], label="Val")
axes[1].set_title("Loss MLP")
axes[1].set_xlabel("Epocas")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=200)
plt.show()

# ============================================================
# 9. EXPORTAR model_weights.h
# ============================================================
def fuse_bn(dense_layer, bn_layer):
    W, b = dense_layer.get_weights()
    gamma, beta, moving_mean, moving_var = bn_layer.get_weights()
    eps = bn_layer.epsilon

    scale = gamma / np.sqrt(moving_var + eps)
    W_fused = W * scale[np.newaxis, :]
    b_fused = (b - moving_mean) * scale + beta
    return W_fused, b_fused


def format_array_1d(name, arr):
    vals = ", ".join(f"{float(v):.6f}f" for v in arr.flatten())
    return f"static const float {name}[{len(arr.flatten())}] = {{{vals}}};\n"


def format_array_2d(name, arr):
    rows, cols = arr.shape
    lines = [f"static const float {name}[{rows}][{cols}] = {{"]
    for i in range(rows):
        vals = ", ".join(f"{float(v):.6f}f" for v in arr[i])
        comma = "," if i < rows - 1 else ""
        lines.append(f"  {{{vals}}}{comma}")
    lines.append("};\n")
    return "\n".join(lines)


print("\nFusionando BatchNorm en pesos Dense...")

W1_f, b1_f = fuse_bn(model.get_layer("dense_1"), model.get_layer("bn_1"))
W2_f, b2_f = fuse_bn(model.get_layer("dense_2"), model.get_layer("bn_2"))
W3_raw, b3_raw = model.get_layer("dense_3").get_weights()
W4_raw, b4_raw = model.get_layer("dense_out").get_weights()

print(f"  W1: {W1_f.shape}, b1: {b1_f.shape}")
print(f"  W2: {W2_f.shape}, b2: {b2_f.shape}")
print(f"  W3: {W3_raw.shape}, b3: {b3_raw.shape}")
print(f"  W4: {W4_raw.shape}, b4: {b4_raw.shape}")

header_path = OUTPUT_DIR / "model_weights.h"
with open(header_path, "w", encoding="utf-8") as f:
    f.write("// AUTO-GENERADO - NO EDITAR MANUALMENTE\n")
    f.write("// Modelo: MLP 3 clases para IDS MQTT (TinyML)\n")
    f.write(f"// Clases: {', '.join(CLASS_NAMES)}\n")
    f.write(f"// Arquitectura: {FEATURE_COUNT}->32->16->8->{NUM_CLASSES}\n")
    f.write("// BatchNorm fusionado en Dense layers\n")
    f.write("#pragma once\n\n")

    f.write(f"static const size_t FEATURE_COUNT = {FEATURE_COUNT};\n")
    f.write(f"static const size_t NUM_CLASSES = {NUM_CLASSES};\n\n")

    f.write("// StandardScaler params\n")
    f.write(format_array_1d("scaler_mean", scaler.mean_))
    f.write(format_array_1d("scaler_std", scaler.scale_))
    f.write("\n")

    f.write(f"// Dense 1 (fused BN): ({FEATURE_COUNT}, 32)\n")
    f.write(format_array_2d("W1_base", W1_f))
    f.write(format_array_1d("b1_base", b1_f))
    f.write("\n")

    f.write("// Dense 2 (fused BN): (32, 16)\n")
    f.write(format_array_2d("W2_base", W2_f))
    f.write(format_array_1d("b2_base", b2_f))
    f.write("\n")

    f.write("// Dense 3: (16, 8)\n")
    f.write(format_array_2d("W3_base", W3_raw))
    f.write(format_array_1d("b3_base", b3_raw))
    f.write("\n")

    f.write(f"// Dense out (softmax): (8, {NUM_CLASSES})\n")
    f.write(format_array_2d("W4_base", W4_raw))
    f.write(format_array_1d("b4_base", b4_raw))

# ============================================================
# 10. EXPORTAR scaler, label_map y modelos
# ============================================================
with open(OUTPUT_DIR / "scaler_params.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "mean": scaler.mean_.tolist(),
            "std": scaler.scale_.tolist(),
            "features": FLOW_FEATURES,
            "input_shape": [FEATURE_COUNT],
            "model_type": "MLP",
            "tensorflow_version": tf.__version__,
            "keras_version": keras.__version__,
        },
        f,
        indent=2,
    )

with open(OUTPUT_DIR / "label_map.json", "w", encoding="utf-8") as f:
    json.dump({name: i for i, name in enumerate(CLASS_NAMES)}, f, indent=2)

keras_path = OUTPUT_DIR / "ids_3class.keras"
h5_path = OUTPUT_DIR / "ids_3class.h5"
weights_path = OUTPUT_DIR / "ids_3class.weights.h5"

model.save(keras_path)
model.save(h5_path, include_optimizer=False)
model.save_weights(weights_path)

print("\nExportacion completa:")
print(" ", header_path)
print(" ", OUTPUT_DIR / "scaler_params.json")
print(" ", OUTPUT_DIR / "label_map.json")
print(" ", keras_path)
print(" ", h5_path)
print(" ", weights_path)
print(" ", OUTPUT_DIR / "confusion_matrix.png")
print(" ", OUTPUT_DIR / "training_curves.png")
print(f"\nAccuracy : {acc:.4f}")
print(f"F1 Score : {f1:.4f}")
