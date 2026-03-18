"""
=============================================================================
 retrain_raw_features.py
 Reentrenamiento del modelo multiclase SIN log1p ni clipping externo.
 La capa Normalization() interna se adapta a los datos CRUDOS,
 así la ESP32 puede enviar features sin transformar.
=============================================================================
 INSTRUCCIONES:
 1. Sube este script al mismo servidor donde tienes los CSVs del dataset.
 2. Ajusta DATA_DIR para que apunte a la carpeta con los .pcap.csv
 3. Corre: python retrain_raw_features.py
 4. Al finalizar, descarga la carpeta export_raw/ que contiene:
    - attack_multiclass.h5
    - attack_multiclass.keras
    - feature_order.csv
    - label_map.json
 5. Corre export_weights.py sobre el nuevo .h5 para generar model_weights.h
=============================================================================
"""

from pathlib import Path
import re, json
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =============================================
# 1) CONFIGURACION - AJUSTA ESTA RUTA
# =============================================
DATA_DIR = Path("/home/sajaimesp/Tesis_Sistemas/Data_Sets/Proyecto_Grado_DataSets")
EXPORT_DIR = Path("export_raw")
EXPORT_DIR.mkdir(exist_ok=True)

CSV_GLOB = "*.pcap.csv"
files = sorted(DATA_DIR.glob(CSV_GLOB))
print(f"DATA_DIR: {DATA_DIR.resolve()}")
print(f"CSV encontrados: {len(files)}")

# =============================================
# 2) ETIQUETADO POR NOMBRE DE ARCHIVO
# =============================================
def infer_label_from_filename(name: str) -> str:
    n = name.lower().replace(".csv", "").replace("&", "_and_")
    n = re.sub(r"[\s]+", "_", n)
    n = n.replace("-", "_")
    n = re.sub(r"_+", "_", n)

    if "dns_spoofing" in n:           return "dns_spoofing"
    if "sqlinjection" in n or "sql_injection" in n: return "sql_injection_uploading"
    if "ddos_ack_fragmentation" in n: return "ddos_ack_fragmentation"
    if "ddos_icmp_flood" in n:        return "ddos_icmp_flood"
    if "ddos_http_flood" in n:        return "ddos_http_flood"
    if "ddos_tcp_flood" in n:         return "ddos_tcp_flood"
    if "dos_http_flood" in n:         return "dos_http_flood"
    if "dos_syn_flood" in n:          return "dos_syn_flood"
    if "dos_tcp_flood" in n:          return "dos_tcp_flood"
    if "benign" in n or "normal" in n: return "benign"
    return None

rows = []
for f in files:
    lab = infer_label_from_filename(f.name)
    rows.append({"file": str(f), "fname": f.name, "label": lab})

df_lab = pd.DataFrame(rows)
df_lab = df_lab.dropna(subset=["label"]).copy()

print("\nDistribución de clases:")
print(df_lab["label"].value_counts())

# =============================================
# 3) CARGA DE DATOS
# =============================================
def reservoir_sample_csv(path, n_rows=30_000, chunksize=200_000, seed=42):
    parts = []
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        chunk = chunk.apply(pd.to_numeric, errors="coerce")
        parts.append(chunk)
        if sum(len(p) for p in parts) >= n_rows:
            break
    out = pd.concat(parts, ignore_index=True)
    if len(out) > n_rows:
        out = out.sample(n=n_rows, random_state=seed).reset_index(drop=True)
    return out

dfs = []
for _, row in df_lab.iterrows():
    fp = Path(row["file"])
    lab = row["label"]
    tmp = reservoir_sample_csv(fp, n_rows=20_000, seed=SEED)
    tmp["label"] = lab
    tmp["__source__"] = fp.name
    dfs.append(tmp)
    
df = pd.concat(dfs, ignore_index=True)
print(f"\nDataset total shape: {df.shape}")
print(df["label"].value_counts().head(15))

# =============================================
# 4) PREPARACION DE FEATURES - SIN LOG1P NI CLIPPING
# =============================================

# IMPORTANTE: Guardar las etiquetas ANTES de cualquier conversión numérica
y_str = df["label"].copy().values
groups = df["__source__"].copy().values

# Verificación de seguridad
print(f"\n🔍 Verificación: y_str primeros 5 = {y_str[:5]}")
print(f"🔍 Verificación: groups primeros 5 = {groups[:5]}")
assert y_str[0] != "nan", "ERROR: y_str contiene 'nan'. Reinicia el kernel y corre solo este script."

# Quitar columnas no-feature
drop_cols = ["label", "__source__"]
X_df = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

# NO incluir features ingeniadas (bytes_per_pkt, syn_ack_ratio, etc.)
# porque la ESP32 no puede calcularlas. Solo quedarnos con las 31 columnas
# originales que vienen del CSV de CICFlowMeter.
engineered_cols = ["bytes_per_pkt", "syn_ack_ratio", "rst_rate", "fin_rate", "tcp_udp_ratio"]
X_df = X_df.drop(columns=[c for c in engineered_cols if c in X_df.columns])
print(f"\n(Columnas ingeniadas eliminadas: {[c for c in engineered_cols if c in df.columns]})")

# Convertir todo a numérico y limpiar
X_df = X_df.apply(pd.to_numeric, errors="coerce")
X_df.replace([np.inf, -np.inf], np.nan, inplace=True)
X_df = X_df.fillna(X_df.median(numeric_only=True))

# =============================================
# *** DIFERENCIA CLAVE CON EL NOTEBOOK ORIGINAL ***
# NO hacemos clipping con percentiles
# NO hacemos log1p()
# Solo VarianceThreshold para quitar columnas constantes
# =============================================
vt = VarianceThreshold(threshold=1e-6)
X_vt = vt.fit_transform(X_df.values)
kept_cols = X_df.columns[vt.get_support()].tolist()

X = X_vt.astype(np.float32)

print(f"\n✅ Features finales: {len(kept_cols)}")
print(f"Columnas: {kept_cols}")

# Debe ser ~31 features, NO 41
if len(kept_cols) > 35:
    print("⚠️ ADVERTENCIA: Más de 35 features detectadas. ¿Corriste add_features() antes?")
    print("   Reinicia el kernel del notebook y corre SOLO las celdas de este script.")

# =============================================
# 5) SPLIT TRAIN / VAL / TEST
# =============================================
labels = sorted(np.unique(y_str))
label_to_id = {l: i for i, l in enumerate(labels)}
y = np.array([label_to_id[v] for v in y_str], dtype=np.int32)

print(f"\nClases: {label_to_id}")

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, test_idx = next(gss.split(X, y, groups=groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
groups_train = groups[train_idx]

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr2_idx, val_idx = next(gss2.split(X_train, y_train, groups=groups_train))
X_tr, X_val = X_train[tr2_idx], X_train[val_idx]
y_tr, y_val = y_train[tr2_idx], y_train[val_idx]

print(f"Train: {X_tr.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

# =============================================
# 6) MODELO - LA CAPA Normalization SE ADAPTA A DATOS CRUDOS
# =============================================
num_classes = len(labels)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_tr),
    y=y_tr
)
class_weight_dict = {int(c): float(w) for c, w in zip(np.unique(y_tr), class_weights)}
print(f"\nclass_weight: {class_weight_dict}")

# La capa Normalization se adapta a los datos CRUDOS (sin log1p)
norm = tf.keras.layers.Normalization(axis=-1)
norm.adapt(X_tr)

inputs = tf.keras.Input(shape=(X_tr.shape[1],), dtype=tf.float32, name="features")
x = norm(inputs)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dense(32, activation="relu")(x)
x = tf.keras.layers.Dropout(0.15)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probs")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc")
    ],
)

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=6,
        restore_best_weights=True
    )
]

history = model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=2048,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# =============================================
# 7) EVALUACION
# =============================================
probs = model.predict(X_test, batch_size=4096)
pred = np.argmax(probs, axis=1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))
print("\nClassification Report:")
print(classification_report(y_test, pred, target_names=labels, digits=4))

# =============================================
# 8) EXPORTAR
# =============================================
# Feature order
pd.Series(kept_cols).to_csv(EXPORT_DIR / "feature_order.csv", index=False, header=False)

# Label map
with open(EXPORT_DIR / "label_map.json", "w", encoding="utf-8") as f:
    json.dump(label_to_id, f, ensure_ascii=False, indent=2)

# Keras format
model.save(EXPORT_DIR / "attack_multiclass.keras")
print(f"✅ .keras guardado en: {(EXPORT_DIR / 'attack_multiclass.keras').resolve()}")

# H5 format
model.save(EXPORT_DIR / "attack_multiclass.h5")
print(f"✅ .h5 guardado en: {(EXPORT_DIR / 'attack_multiclass.h5').resolve()}")

# =============================================
# 9) VERIFICACION RAPIDA CON FEATURES CRUDAS
# =============================================
# Simula exactamente los mismos datos "dummy" que el ESP32 envía
print("\n--- Verificación con features crudas (como las enviaría el ESP32) ---")
dummy_benign = np.zeros((1, len(kept_cols)), dtype=np.float32)
# Valores típicos de tráfico benigno:
feature_map = {name: i for i, name in enumerate(kept_cols)}

if "Header_Length" in feature_map: dummy_benign[0, feature_map["Header_Length"]] = 64.0
if "Protocol Type" in feature_map: dummy_benign[0, feature_map["Protocol Type"]] = 6.0
if "Time_To_Live" in feature_map:  dummy_benign[0, feature_map["Time_To_Live"]] = 64.0
if "Rate" in feature_map:          dummy_benign[0, feature_map["Rate"]] = 25.0
if "ack_flag_number" in feature_map: dummy_benign[0, feature_map["ack_flag_number"]] = 1.0
if "ack_count" in feature_map:     dummy_benign[0, feature_map["ack_count"]] = 8.0
if "TCP" in feature_map:           dummy_benign[0, feature_map["TCP"]] = 1.0
if "Tot sum" in feature_map:       dummy_benign[0, feature_map["Tot sum"]] = 512.0
if "Min" in feature_map:           dummy_benign[0, feature_map["Min"]] = 64.0
if "Max" in feature_map:           dummy_benign[0, feature_map["Max"]] = 64.0
if "AVG" in feature_map:           dummy_benign[0, feature_map["AVG"]] = 64.0
if "Tot size" in feature_map:      dummy_benign[0, feature_map["Tot size"]] = 512.0
if "IAT" in feature_map:           dummy_benign[0, feature_map["IAT"]] = 20.0
if "Number" in feature_map:        dummy_benign[0, feature_map["Number"]] = 8.0

prob_benign = model.predict(dummy_benign)
pred_class = np.argmax(prob_benign, axis=1)[0]
id_to_label = {i: l for l, i in label_to_id.items()}
print(f"  Predicción con features crudas benign: {id_to_label[pred_class]} (clase {pred_class})")
print(f"  Probabilidades: {prob_benign[0]}")

print("\n✅ Listo. Ahora corre export_weights.py sobre el nuevo .h5 para generar model_weights.h")
