"""
export_weights_rpi.py — Exporta SOLO los pesos del modelo CNN-1D
================================================================
Evita incompatibilidades de versión entre Keras del PC y la Raspberry Pi.
La arquitectura se reconstruye en cada dispositivo con build_cnn_model(),
y solo se transfieren los pesos numéricos (universalmente compatibles).

Uso:
  1. Corre este script en el PC: python export_weights_rpi.py
  2. Copia "ids_3class_cnn.weights.h5" a la Raspberry Pi (~/)
  3. El gateway_hfl.py ya lo carga automáticamente vía load_base_cnn_model()
"""
import tensorflow as tf
import numpy as np
from pathlib import Path

print(f"[PC] TF: {tf.__version__}  |  Keras: {tf.keras.__version__}")

# ── Arquitectura IDÉNTICA a build_cnn_model() del gateway ──
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(13, 1), name='input_features'),
    tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding='same', name='conv_1'),
    tf.keras.layers.BatchNormalization(name='bn_conv_1'),
    tf.keras.layers.Dropout(0.3, name='drop_1'),
    tf.keras.layers.Conv1D(16, kernel_size=3, activation='relu', padding='same', name='conv_2'),
    tf.keras.layers.BatchNormalization(name='bn_conv_2'),
    tf.keras.layers.Dropout(0.2, name='drop_2'),
    tf.keras.layers.GlobalAveragePooling1D(name='gap'),
    tf.keras.layers.Dense(8, activation='relu', name='dense_1'),
    tf.keras.layers.Dense(3, activation='softmax', name='dense_out'),
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
model.build(input_shape=(None, 13, 1))

# ── Opción A: intentar cargar pesos del .keras existente ──
source_keras = Path("ids_3class.keras")
if source_keras.exists():
    print(f"\n[INFO] Encontrado {source_keras}. Intentando extraer pesos...")
    try:
        src = tf.keras.models.load_model(str(source_keras), compile=False)
        # Copiar capa por capa por nombre (robusto ante versiones distintas)
        loaded = 0
        for layer in src.layers:
            try:
                target_layer = model.get_layer(layer.name)
                target_layer.set_weights(layer.get_weights())
                loaded += 1
                print(f"  ✓ Pesos copiados: {layer.name}")
            except Exception:
                pass  # Capa no existe en arquitectura destino — OK
        print(f"[INFO] {loaded} capas copiadas del modelo fuente.")
    except Exception as e:
        print(f"[WARN] No se pudo cargar {source_keras}: {e}")
        print("[INFO] Se exportarán pesos iniciales aleatorios (el modelo aprenderá en FL).")
else:
    print(f"[INFO] {source_keras} no encontrado. Exportando pesos iniciales aleatorios.")

# ── Exportar SOLO pesos (formato agnóstico de versión) ──
out_path = Path("ids_3class_cnn.weights.h5")
model.save_weights(str(out_path))
print(f"\n[OK] Pesos exportados: {out_path}  ({out_path.stat().st_size // 1024} KB)")
print("     → Copia este archivo a la Raspberry Pi (~/) y reinicia el gateway.")

# ── Verificación rápida ──
model2 = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(13, 1), name='input_features'),
    tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding='same', name='conv_1'),
    tf.keras.layers.BatchNormalization(name='bn_conv_1'),
    tf.keras.layers.Dropout(0.3, name='drop_1'),
    tf.keras.layers.Conv1D(16, kernel_size=3, activation='relu', padding='same', name='conv_2'),
    tf.keras.layers.BatchNormalization(name='bn_conv_2'),
    tf.keras.layers.Dropout(0.2, name='drop_2'),
    tf.keras.layers.GlobalAveragePooling1D(name='gap'),
    tf.keras.layers.Dense(8, activation='relu', name='dense_1'),
    tf.keras.layers.Dense(3, activation='softmax', name='dense_out'),
])
model2.build(input_shape=(None, 13, 1))
model2.load_weights(str(out_path))

x_test = np.random.rand(1, 13, 1).astype(np.float32)
pred = model2.predict(x_test, verbose=0)
print(f"[VERIFY] Predicción de prueba: {pred}  → suma={pred.sum():.4f} (debe ser ~1.0)")
print("\n✅ Archivo listo para copiar a la Raspberry Pi.")
