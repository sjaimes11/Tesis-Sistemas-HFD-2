# -*- coding: utf-8 -*-
"""
=============================================================================
 train_3class.py — Entrenamiento MLP 3 clases para ESP32 TinyML IDS
=============================================================================
 Clases:
   0 = normal
   1 = mqtt_bruteforce
   2 = scan_A

 Arquitectura: 13 -> 32 -> BN -> 16 -> BN -> 8 -> 3 (softmax)
 Export: model_weights.h con BatchNorm fusionado en los pesos Dense

 Ejecutar:
   pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
   python train_3class.py
=============================================================================
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score
)

SEED = 42
np.random.seed(SEED)

# ============================================================
# 1. CARGAR DATOS
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")

print("Cargando datos...")
df_normal    = pd.read_csv(os.path.join(DATA_DIR, "uniflow_normal.csv"), low_memory=False)
df_bruteforce = pd.read_csv(os.path.join(DATA_DIR, "uniflow_mqtt_bruteforce.csv"), low_memory=False)
df_scan      = pd.read_csv(os.path.join(DATA_DIR, "uniflow_scan_A.csv"), low_memory=False)

# Asignar etiqueta de clase según el archivo de origen
df_normal["class_label"]     = 0
df_bruteforce["class_label"] = 1
df_scan["class_label"]       = 2

print(f"  Normal:         {len(df_normal)}")
print(f"  MQTT Bruteforce: {len(df_bruteforce)}")
print(f"  Scan A:          {len(df_scan)}")

# ============================================================
# 2. FEATURES
# ============================================================
FLOW_FEATURES = [
    "num_pkts", "mean_iat", "std_iat", "min_iat", "max_iat",
    "mean_pkt_len", "num_bytes", "num_psh_flags", "num_rst_flags",
    "num_urg_flags", "std_pkt_len", "min_pkt_len", "max_pkt_len",
]
TARGET = "class_label"
CLASS_NAMES = ["normal", "mqtt_bruteforce", "scan_A"]
NUM_CLASSES = 3

# ============================================================
# 3. BALANCEAR CLASES (undersample a la clase minoritaria)
# ============================================================
min_count = min(len(df_normal), len(df_bruteforce), len(df_scan))
print(f"\nBalanceando clases a {min_count} muestras cada una...")

df_normal_s    = df_normal.sample(n=min_count, random_state=SEED)
df_bruteforce_s = df_bruteforce.sample(n=min_count, random_state=SEED)
df_scan_s      = df_scan.sample(n=min_count, random_state=SEED)

df = pd.concat([df_normal_s, df_bruteforce_s, df_scan_s], ignore_index=True)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# Limpiar
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)

print(f"Dataset final: {df.shape}")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name}: {(df[TARGET] == i).sum()}")

# ============================================================
# 4. SPLIT Y ESCALAR
# ============================================================
X = df[FLOW_FEATURES].values
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\nTrain: {X_train_sc.shape}  |  Test: {X_test_sc.shape}")

# ============================================================
# 5. ENTRENAR MODELO KERAS
# ============================================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

print(f"\nTensorFlow: {tf.__version__}")
print(f"Arquitectura: {len(FLOW_FEATURES)} -> 32 -> 16 -> 8 -> {NUM_CLASSES}")

model = keras.Sequential([
    layers.Input(shape=(len(FLOW_FEATURES),)),
    layers.Dense(32, activation="relu", name="dense_1"),
    layers.BatchNormalization(name="bn_1"),
    layers.Dropout(0.3),
    layers.Dense(16, activation="relu", name="dense_2"),
    layers.BatchNormalization(name="bn_2"),
    layers.Dropout(0.2),
    layers.Dense(8, activation="relu", name="dense_3"),
    layers.Dense(NUM_CLASSES, activation="softmax", name="dense_out"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
reduce_lr  = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)

history = model.fit(
    X_train_sc, y_train,
    validation_split=0.15,
    epochs=100,
    batch_size=256,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)

# ============================================================
# 6. EVALUACION
# ============================================================
y_pred = model.predict(X_test_sc, verbose=0).argmax(axis=1)

acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average="weighted")

print(f"\nAccuracy : {acc:.4f}")
print(f"F1 (weighted): {f1:.4f}\n")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix - 3-Class IDS")
plt.ylabel("Real")
plt.xlabel("Predicho")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "confusion_matrix.png"))
print("confusion_matrix.png guardada.")

# ============================================================
# 7. EXPORTAR model_weights.h (BatchNorm fusionado)
# ============================================================
def fuse_bn(dense_layer, bn_layer):
    """Fusiona BatchNorm en los pesos de la capa Dense precedente."""
    W, b = dense_layer.get_weights()
    gamma, beta, moving_mean, moving_var = bn_layer.get_weights()
    eps = bn_layer.epsilon

    # Factor de escala del BatchNorm
    scale = gamma / np.sqrt(moving_var + eps)

    # Fusionar en W y b
    W_fused = W * scale[np.newaxis, :]
    b_fused = (b - moving_mean) * scale + beta

    return W_fused, b_fused

print("\nFusionando BatchNorm en pesos Dense...")

# Capas con BN
W1_f, b1_f = fuse_bn(model.get_layer("dense_1"), model.get_layer("bn_1"))
W2_f, b2_f = fuse_bn(model.get_layer("dense_2"), model.get_layer("bn_2"))

# Capas sin BN
W3_raw, b3_raw = model.get_layer("dense_3").get_weights()
W4_raw, b4_raw = model.get_layer("dense_out").get_weights()

print(f"  W1: {W1_f.shape}, b1: {b1_f.shape}")
print(f"  W2: {W2_f.shape}, b2: {b2_f.shape}")
print(f"  W3: {W3_raw.shape}, b3: {b3_raw.shape}")
print(f"  W4: {W4_raw.shape}, b4: {b4_raw.shape}")

def format_array_1d(name, arr):
    vals = ", ".join(f"{v:.6f}f" for v in arr.flatten())
    return f"static const float {name}[{len(arr)}] = {{{vals}}};\n"

def format_array_2d(name, arr):
    rows, cols = arr.shape
    lines = [f"static const float {name}[{rows}][{cols}] = {{"]
    for i in range(rows):
        vals = ", ".join(f"{v:.6f}f" for v in arr[i])
        comma = "," if i < rows - 1 else ""
        lines.append(f"  {{{vals}}}{comma}")
    lines.append("};\n")
    return "\n".join(lines)

header_path = os.path.join(os.path.dirname(__file__), "model_weights.h")
with open(header_path, "w") as f:
    f.write("// AUTO-GENERADO — NO EDITAR MANUALMENTE\n")
    f.write(f"// Modelo: MLP 3 clases para IDS MQTT (TinyML)\n")
    f.write(f"// Clases: {', '.join(CLASS_NAMES)}\n")
    f.write(f"// Arquitectura: {len(FLOW_FEATURES)}->32->16->8->{NUM_CLASSES}\n")
    f.write(f"// BatchNorm fusionado en Dense layers\n")
    f.write("#pragma once\n\n")

    f.write(f"static const size_t FEATURE_COUNT = {len(FLOW_FEATURES)};\n")
    f.write(f"static const size_t NUM_CLASSES = {NUM_CLASSES};\n\n")

    # Scaler
    f.write(format_array_1d("scaler_mean", scaler.mean_))
    f.write(format_array_1d("scaler_std", scaler.scale_))
    f.write("\n")

    # Layer 1 (fused with BN)
    f.write(f"// Dense 1 (fused BN): ({len(FLOW_FEATURES)}, 32)\n")
    f.write(format_array_2d("W1_base", W1_f))
    f.write(format_array_1d("b1_base", b1_f))
    f.write("\n")

    # Layer 2 (fused with BN)
    f.write(f"// Dense 2 (fused BN): (32, 16)\n")
    f.write(format_array_2d("W2_base", W2_f))
    f.write(format_array_1d("b2_base", b2_f))
    f.write("\n")

    # Layer 3 (no BN)
    f.write(f"// Dense 3: (16, 8)\n")
    f.write(format_array_2d("W3_base", W3_raw))
    f.write(format_array_1d("b3_base", b3_raw))
    f.write("\n")

    # Layer 4 (output, softmax)
    f.write(f"// Dense out (softmax): (8, {NUM_CLASSES})\n")
    f.write(format_array_2d("W4_base", W4_raw))
    f.write(format_array_1d("b4_base", b4_raw))

print(f"\nmodel_weights.h guardado en: {header_path}")

# ============================================================
# 8. EXPORTAR scaler y label_map
# ============================================================
with open(os.path.join(os.path.dirname(__file__), "scaler_params.json"), "w") as f:
    json.dump({
        "mean": scaler.mean_.tolist(),
        "std": scaler.scale_.tolist(),
        "features": FLOW_FEATURES,
    }, f, indent=2)

with open(os.path.join(os.path.dirname(__file__), "label_map.json"), "w") as f:
    json.dump({name: i for i, name in enumerate(CLASS_NAMES)}, f, indent=2)

# Guardar modelo Keras
model.save(os.path.join(os.path.dirname(__file__), "ids_3class.keras"))

print("\nExportación completa:")
print(f"  model_weights.h   - C header para ESP32")
print(f"  scaler_params.json")
print(f"  label_map.json")
print(f"  ids_3class.keras")
print(f"  confusion_matrix.png")
print(f"\nAccuracy: {acc:.4f} | F1: {f1:.4f}")
