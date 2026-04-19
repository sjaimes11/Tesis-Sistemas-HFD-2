# HFL v7 - K-Nearest Neighbors

## Esquema teacher -> student

- Teacher original: `K-Nearest Neighbors`
- Student desplegable: MLP `13 -> 32 -> 16 -> 8 -> 3`
- Formato del student:
  - `ids_3class.h5`
  - `ids_3class.keras` (opcional)
- Export TinyML: `model_weights.h`

## Por que asi

Random Forest, KNN y Naive Bayes no encajan nativamente en el protocolo
de intercambio de pesos de HFL v7. Por eso aqui se usa un teacher clasico
para entrenar una red estudiante compatible con tu arquitectura actual.

## Resultado mas reciente

- Teacher accuracy: 0.9121
- Student accuracy: 0.8998
- Teacher F1 macro: 0.9124
- Student F1 macro: 0.8985

## Artefactos

- `ids_3class.h5`: student model compatible con entrenamiento posterior.
- `ids_3class.keras`: version nativa opcional.
- `model_weights.h`: pesos base para ESP32.
- `teacher_knn.joblib`: teacher clasico original.
- `training_summary.json`: metricas y configuracion del entrenamiento.
