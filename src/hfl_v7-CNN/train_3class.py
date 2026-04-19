# -*- coding: utf-8 -*-
"""
=============================================================================
 train_3class.py — Entrenamiento CNN-1D 3 clases para ESP32 TinyML IDS
=============================================================================
 Alternativa al MLP (hfl_v7-RN) usando una Red Neuronal Convolucional 1D.
 La CNN-1D trata las 13 features como una "secuencia temporal" de longitud 13
 y aplica filtros convolucionales para detectar patrones locales entre
 features adyacentes (ej: correlación entre num_pkts y mean_iat).

 Clases:
   0 = normal
   1 = mqtt_bruteforce
   2 = scan_A

 Arquitectura:
   Input (13,1)
   -> Conv1D(32 filtros, kernel=3, relu) -> BN -> Dropout(0.3)
   -> Conv1D(16 filtros, kernel=3, relu) -> BN -> Dropout(0.2)
   -> GlobalAvgPooling1D
   -> Dense(8, relu)       <- W_dense1 (capa federada)
   -> Dense(3, softmax)    <- W_dense2 (capa federada)

 Capas FL (federadas): dense_1 (8 neuronas) + dense_out (3 neuronas)
 Export: model_weights.h con pesos de las capas Dense (para TinyML ESP32)
         Las capas Conv se exportan por separado como referencia.

 NOTA: La inferencia en ESP32 con capas Conv1D es más costosa que MLP puro.
       Este modelo es ideal para comparación académica en Fog (Raspberry Pi).
       Para el Edge (ESP32), el MLP de hfl_v7-RN sigue siendo preferible.

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
df_normal     = pd.read_csv(os.path.join(DATA_DIR, "uniflow_normal.csv"), low_memory=False)
df_bruteforce = pd.read_csv(os.path.join(DATA_DIR, "uniflow_mqtt_bruteforce.csv"), low_memory=False)
df_scan       = pd.read_csv(os.path.join(DATA_DIR, "uniflow_scan_A.csv"), low_memory=False)

df_normal["class_label"]     = 0
df_bruteforce["class_label"] = 1
df_scan["class_label"]       = 2

print(f"  Normal:          {len(df_normal)}")
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
FEATURE_COUNT = len(FLOW_FEATURES)  # 13

# ============================================================
# 3. BALANCEAR CLASES (undersample a la clase minoritaria)
# ============================================================
min_count = min(len(df_normal), len(df_bruteforce), len(df_scan))
print(f"\nBalanceando clases a {min_count} muestras cada una...")

df_normal_s     = df_normal.sample(n=min_count, random_state=SEED)
df_bruteforce_s = df_bruteforce.sample(n=min_count, random_state=SEED)
df_scan_s       = df_scan.sample(n=min_count, random_state=SEED)

df = pd.concat([df_normal_s, df_bruteforce_s, df_scan_s], ignore_index=True)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# Limpiar valores infinitos / NaN
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

# ── CNN-1D requiere entrada de forma (batch, timesteps, channels)
# Reinterpretamos las 13 features como 13 "pasos" con 1 canal cada uno
X_train_cnn = X_train_sc.reshape(-1, FEATURE_COUNT, 1)  # (N, 13, 1)
X_test_cnn  = X_test_sc.reshape(-1, FEATURE_COUNT, 1)

print(f"\nTrain: {X_train_cnn.shape}  |  Test: {X_test_cnn.shape}")

# ============================================================
# 5. CONSTRUIR MODELO CNN-1D
# ============================================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

print(f"\nTensorFlow: {tf.__version__}")
print(f"Arquitectura CNN-1D: ({FEATURE_COUNT},1) -> Conv32 -> BN -> Conv16 -> BN -> GAP -> Dense8 -> Dense{NUM_CLASSES}")

model = keras.Sequential([
    # Entrada: (13, 1)
    layers.Input(shape=(FEATURE_COUNT, 1)),

    # Bloque Conv 1
    layers.Conv1D(filters=32, kernel_size=3, activation="relu",
                  padding="same", name="conv_1"),
    layers.BatchNormalization(name="bn_conv_1"),
    layers.Dropout(0.3, name="drop_1"),

    # Bloque Conv 2
    layers.Conv1D(filters=16, kernel_size=3, activation="relu",
                  padding="same", name="conv_2"),
    layers.BatchNormalization(name="bn_conv_2"),
    layers.Dropout(0.2, name="drop_2"),

    # Pooling global — colapsa la dimensión temporal a un vector fijo
    layers.GlobalAveragePooling1D(name="gap"),

    # Capas Dense (estas son las que viajan en FL igual que en el MLP)
    layers.Dense(8, activation="relu", name="dense_1"),
    layers.Dense(NUM_CLASSES, activation="softmax", name="dense_out"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ============================================================
# 6. ENTRENAR
# ============================================================
early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
reduce_lr  = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)

history = model.fit(
    X_train_cnn, y_train,
    validation_split=0.15,
    epochs=100,
    batch_size=256,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)

# ============================================================
# 7. EVALUACIÓN
# ============================================================
y_pred = model.predict(X_test_cnn, verbose=0).argmax(axis=1)

acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average="weighted")

print(f"\nAccuracy : {acc:.4f}")
print(f"F1 (weighted): {f1:.4f}\n")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix - CNN-1D 3-Class IDS")
plt.ylabel("Real")
plt.xlabel("Predicho")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "confusion_matrix.png"))
print("confusion_matrix.png guardada.")

# Curvas de entrenamiento
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history["accuracy"], label="Train")
axes[0].plot(history.history["val_accuracy"], label="Val")
axes[0].set_title("Accuracy CNN-1D")
axes[0].set_xlabel("Épocas"); axes[0].set_ylabel("Accuracy")
axes[0].legend(); axes[0].grid(True)

axes[1].plot(history.history["loss"], label="Train")
axes[1].plot(history.history["val_loss"], label="Val")
axes[1].set_title("Loss CNN-1D")
axes[1].set_xlabel("Épocas"); axes[1].set_ylabel("Loss")
axes[1].legend(); axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "training_curves.png"))
print("training_curves.png guardada.")

# ============================================================
# 8. EXPORTAR model_weights.h
# ============================================================
# ── Estrategia de exportación para TinyML:
#    Las capas Conv1D se fusionan con sus BN y se exportan.
#    Las capas Dense (dense_1, dense_out) se exportan directamente.
#    El firmware C++ deberá implementar:
#      1. Conv1D con padding="same" (más costoso que MLP)
#      2. GlobalAveragePooling1D
#      3. Dense + ReLU / Softmax
#
# ── Capas FL (federadas via MQTT/HTTP igual que en hfl_v7-RN):
#    dense_1  (shape: 16 -> 8)
#    dense_out (shape: 8  -> 3)
# ────────────────────────────────────────────────────────────

def fuse_bn_conv(conv_layer, bn_layer):
    """
    Fusiona BatchNorm en los pesos de una capa Conv1D.
    Conv1D weights shape: (kernel_size, in_channels, out_filters)
    BN params: gamma, beta, moving_mean, moving_var
    """
    W, b = conv_layer.get_weights()          # (kernel, in_ch, filters)
    gamma, beta, moving_mean, moving_var = bn_layer.get_weights()
    eps = bn_layer.epsilon

    scale = gamma / np.sqrt(moving_var + eps)  # (filters,)

    # Escalar filtros y bias
    W_fused = W * scale[np.newaxis, np.newaxis, :]  # broadcast sobre (k, in_ch, filters)
    b_fused = (b - moving_mean) * scale + beta

    return W_fused, b_fused


print("\nFusionando BatchNorm en pesos Conv1D y Dense...")

W_conv1_f, b_conv1_f = fuse_bn_conv(model.get_layer("conv_1"), model.get_layer("bn_conv_1"))
W_conv2_f, b_conv2_f = fuse_bn_conv(model.get_layer("conv_2"), model.get_layer("bn_conv_2"))

W_dense1, b_dense1 = model.get_layer("dense_1").get_weights()
W_dense_out, b_dense_out = model.get_layer("dense_out").get_weights()

print(f"  Conv1 (fused BN): kernel={W_conv1_f.shape}, bias={b_conv1_f.shape}")
print(f"  Conv2 (fused BN): kernel={W_conv2_f.shape}, bias={b_conv2_f.shape}")
print(f"  Dense1:           W={W_dense1.shape}, b={b_dense1.shape}")
print(f"  Dense_out:        W={W_dense_out.shape}, b={b_dense_out.shape}")


def format_array_1d(name, arr):
    vals = ", ".join(f"{v:.6f}f" for v in arr.flatten())
    return f"static const float {name}[{len(arr.flatten())}] = {{{vals}}};\n"


def format_array_2d(name, arr):
    """Formatea array 2D (filas x columnas) como C array 2D."""
    rows, cols = arr.shape
    lines = [f"static const float {name}[{rows}][{cols}] = {{"]
    for i in range(rows):
        vals = ", ".join(f"{v:.6f}f" for v in arr[i])
        comma = "," if i < rows - 1 else ""
        lines.append(f"  {{{vals}}}{comma}")
    lines.append("};\n")
    return "\n".join(lines)


def format_array_3d(name, arr):
    """
    Formatea array 3D (kernel_size x in_channels x out_filters) como C array 3D.
    Para Conv1D: (kernel, in_ch, filters)
    """
    k, in_ch, filters = arr.shape
    lines = [f"static const float {name}[{k}][{in_ch}][{filters}] = {{"]
    for ki in range(k):
        lines.append("  {")
        for ic in range(in_ch):
            vals = ", ".join(f"{v:.6f}f" for v in arr[ki, ic])
            comma_ic = "," if ic < in_ch - 1 else ""
            lines.append(f"    {{{vals}}}{comma_ic}")
        comma_k = "," if ki < k - 1 else ""
        lines.append(f"  }}{comma_k}")
    lines.append("};\n")
    return "\n".join(lines)


header_path = os.path.join(os.path.dirname(__file__), "model_weights.h")
with open(header_path, "w") as f:
    f.write("// AUTO-GENERADO — NO EDITAR MANUALMENTE\n")
    f.write(f"// Modelo: CNN-1D 3 clases para IDS MQTT (TinyML)\n")
    f.write(f"// Clases: {', '.join(CLASS_NAMES)}\n")
    f.write(f"// Arquitectura: ({FEATURE_COUNT},1)->Conv32->BN->Conv16->BN->GAP->Dense8->Dense{NUM_CLASSES}\n")
    f.write(f"// BatchNorm fusionado en Conv layers\n")
    f.write(f"// Capas FL: dense_1 + dense_out\n")
    f.write("#pragma once\n\n")

    f.write(f"static const size_t FEATURE_COUNT = {FEATURE_COUNT};\n")
    f.write(f"static const size_t NUM_CLASSES = {NUM_CLASSES};\n\n")

    # Scaler
    f.write("// StandardScaler params\n")
    f.write(format_array_1d("scaler_mean", scaler.mean_))
    f.write(format_array_1d("scaler_std", scaler.scale_))
    f.write("\n")

    # Conv 1 (kernel_size=3, in=1, out=32)
    f.write(f"// Conv1D_1 (fused BN): kernel (3, 1, 32)\n")
    f.write(format_array_3d("W_conv1", W_conv1_f))
    f.write(format_array_1d("b_conv1", b_conv1_f))
    f.write("\n")

    # Conv 2 (kernel_size=3, in=32, out=16)
    f.write(f"// Conv1D_2 (fused BN): kernel (3, 32, 16)\n")
    f.write(format_array_3d("W_conv2", W_conv2_f))
    f.write(format_array_1d("b_conv2", b_conv2_f))
    f.write("\n")

    # Dense 1 — Capa federada equivalente a W3 en MLP
    gap_out_size = 16  # = número de filtros de conv_2
    f.write(f"// Dense1 (federada — equiv. W3 en MLP): ({gap_out_size}, 8)\n")
    f.write(format_array_2d("W_dense1", W_dense1))
    f.write(format_array_1d("b_dense1", b_dense1))
    f.write("\n")

    # Dense out — Capa federada equivalente a W4 en MLP
    f.write(f"// Dense_out (federada — equiv. W4 en MLP): (8, {NUM_CLASSES})\n")
    f.write(format_array_2d("W_dense_out", W_dense_out))
    f.write(format_array_1d("b_dense_out", b_dense_out))

print(f"\nmodel_weights.h guardado en: {header_path}")

# ============================================================
# 9. EXPORTAR scaler, label_map y modelo Keras
# ============================================================
with open(os.path.join(os.path.dirname(__file__), "scaler_params.json"), "w") as f:
    json.dump({
        "mean": scaler.mean_.tolist(),
        "std": scaler.scale_.tolist(),
        "features": FLOW_FEATURES,
        "input_shape": [FEATURE_COUNT, 1],
        "model_type": "CNN-1D",
    }, f, indent=2)

with open(os.path.join(os.path.dirname(__file__), "label_map.json"), "w") as f:
    json.dump({name: i for i, name in enumerate(CLASS_NAMES)}, f, indent=2)

model.save(os.path.join(os.path.dirname(__file__), "ids_3class.keras"))

print("\nExportación completa:")
print(f"  model_weights.h     - C header para ESP32 (Conv1D + Dense)")
print(f"  scaler_params.json  - Parámetros de normalización")
print(f"  label_map.json      - Mapa de etiquetas")
print(f"  ids_3class.keras    - Modelo Keras completo (para gateway)")
print(f"  confusion_matrix.png")
print(f"  training_curves.png")
print(f"\n{'='*50}")
print(f"  Accuracy : {acc:.4f}")
print(f"  F1 Score : {f1:.4f}")
print(f"{'='*50}")
