"""
=============================================================================
 model_B_deeper.py — MLP PROFUNDO: 64 → 32 → 16 → 6
 Hipótesis: más capas aprenden representaciones jerárquicas de las features.
 Capas más estrechas fuerzan al modelo a comprimir y generalizar.
=============================================================================
"""
from pathlib import Path
import re, json, numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

SEED = 42; np.random.seed(SEED); tf.random.set_seed(SEED)
MODEL_NAME = "Model_B_Deeper_64_32_16"

DATA_DIR = Path("/home/sajaimesp/Tesis_Sistemas/Data_Sets/Proyecto_Grado_DataSets")
EXPORT_DIR = Path("export_modelB"); EXPORT_DIR.mkdir(exist_ok=True)
KEEP_CLASSES = ["benign","ddos_ack_fragmentation","ddos_icmp_flood","ddos_tcp_flood","dos_syn_flood","dos_tcp_flood"]

# ====================== DATA LOADING (compartido) ======================
def infer_label(name):
    n = re.sub(r"_+","_", re.sub(r"[\s]+","_", name.lower().replace(".csv","").replace("&","_and_")).replace("-","_"))
    for k,v in [("dns_spoofing","dns_spoofing"),("sqlinjection","sql_injection_uploading"),("sql_injection","sql_injection_uploading"),
                ("ddos_ack_fragmentation","ddos_ack_fragmentation"),("ddos_icmp_flood","ddos_icmp_flood"),("ddos_http_flood","ddos_http_flood"),
                ("ddos_tcp_flood","ddos_tcp_flood"),("dos_http_flood","dos_http_flood"),("dos_syn_flood","dos_syn_flood"),
                ("dos_tcp_flood","dos_tcp_flood"),("benign","benign"),("normal","benign")]:
        if k in n: return v
    return None

def load_csv(path, n=20000):
    parts = []
    for chunk in pd.read_csv(path, chunksize=200000, low_memory=False):
        chunk = chunk.apply(pd.to_numeric, errors="coerce"); parts.append(chunk)
        if sum(len(p) for p in parts) >= n: break
    out = pd.concat(parts, ignore_index=True)
    return out.sample(n=n, random_state=SEED).reset_index(drop=True) if len(out)>n else out

files = sorted(DATA_DIR.glob("*.pcap.csv"))
df_lab = pd.DataFrame([{"file":str(f),"fname":f.name,"label":infer_label(f.name)} for f in files]).dropna(subset=["label"])
df_lab = df_lab[df_lab["label"].isin(KEEP_CLASSES)].copy()

dfs = []
for _,row in df_lab.iterrows():
    tmp = load_csv(row["file"]); tmp["label"]=row["label"]; tmp["__source__"]=Path(row["file"]).name; dfs.append(tmp)
df = pd.concat(dfs, ignore_index=True)

y_str = df["label"].copy().values; groups = df["__source__"].copy().values
X_df = df.drop(columns=[c for c in ["label","__source__"] if c in df.columns]).copy()
for c in ["bytes_per_pkt","syn_ack_ratio","rst_rate","fin_rate","tcp_udp_ratio"]:
    if c in X_df.columns: X_df.drop(columns=c, inplace=True)
X_df = X_df.apply(pd.to_numeric, errors="coerce")
X_df.replace([np.inf,-np.inf], np.nan, inplace=True)
X_df.fillna(X_df.median(numeric_only=True), inplace=True)

vt = VarianceThreshold(threshold=1e-6); X = vt.fit_transform(X_df.values).astype(np.float32)
kept_cols = X_df.columns[vt.get_support()].tolist()
print(f"Features: {len(kept_cols)}")

labels = sorted(np.unique(y_str)); label_to_id = {l:i for i,l in enumerate(labels)}
id_to_label = {i:l for l,i in label_to_id.items()}
y = np.array([label_to_id[v] for v in y_str], dtype=np.int32); num_classes = len(labels)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr_idx, te_idx = next(gss.split(X,y,groups=groups))
X_train, X_test, y_train, y_test = X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr2, va = next(gss2.split(X_train, y_train, groups=groups[tr_idx]))
X_tr, X_val, y_tr, y_val = X_train[tr2], X_train[va], y_train[tr2], y_train[va]
print(f"Train:{X_tr.shape} Val:{X_val.shape} Test:{X_test.shape}")

cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
cw_dict = {int(c):float(w) for c,w in zip(np.unique(y_tr), cw)}

# ====================== MODELO B: DEEPER 64->32->16->6 ======================
print(f"\n{'='*50}\n {MODEL_NAME}\n MLP: {X_tr.shape[1]} -> 64 -> 32 -> 16 -> {num_classes}\n{'='*50}")

norm = tf.keras.layers.Normalization(axis=-1); norm.adapt(X_tr)
inp = tf.keras.Input(shape=(X_tr.shape[1],), dtype=tf.float32, name="features")
x = norm(inp)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(32, activation="relu")(x)
x = tf.keras.layers.Dropout(0.15)(x)
x = tf.keras.layers.Dense(16, activation="relu")(x)
x = tf.keras.layers.Dropout(0.1)(x)
out = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probs")(x)
model = tf.keras.Model(inp, out)

model.compile(optimizer=tf.keras.optimizers.Adam(5e-4), loss="sparse_categorical_crossentropy",
              metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc")])
model.summary()

hist = model.fit(X_tr, y_tr, validation_data=(X_val,y_val), epochs=80, batch_size=1024,
                 callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",patience=10,restore_best_weights=True)],
                 class_weight=cw_dict, verbose=1)

# ====================== EVALUACION ======================
pred = np.argmax(model.predict(X_test, batch_size=4096), axis=1)
print(f"\n{'='*50} RESULTADOS {MODEL_NAME} {'='*50}")
print(confusion_matrix(y_test, pred, labels=range(len(labels))))
print(classification_report(y_test, pred, labels=range(len(labels)), target_names=labels, digits=4, zero_division=0))

# ====================== EXPORTAR ======================
pd.Series(kept_cols).to_csv(EXPORT_DIR/"feature_order.csv", index=False, header=False)
with open(EXPORT_DIR/"label_map.json","w") as f: json.dump(label_to_id,f,indent=2)
model.save(EXPORT_DIR/"attack_multiclass.keras"); model.save(EXPORT_DIR/"attack_multiclass.h5")

# ====================== VERIFICACION DUMMY ======================
feature_map = {n:i for i,n in enumerate(kept_cols)}
dummy = np.zeros((1,len(kept_cols)), dtype=np.float32)
for feat,val in [("Header_Length",64),("Protocol Type",6),("Time_To_Live",64),("Rate",25),
                 ("ack_flag_number",1),("ack_count",8),("TCP",1),("IPv",1),("Tot sum",512),
                 ("Min",64),("Max",64),("AVG",64),("Tot size",512),("IAT",20),("Number",8)]:
    if feat in feature_map: dummy[0,feature_map[feat]]=val
p = model.predict(dummy); c = np.argmax(p,axis=1)[0]
print(f"\n🧪 Dummy benigno → {id_to_label[c]} (confianza {p[0][c]*100:.1f}%)")
print(f"   Probabilidades: {dict(zip(labels,[f'{v:.3f}' for v in p[0]]))}")
print(f"\n✅ {MODEL_NAME} exportado en {EXPORT_DIR}/")
