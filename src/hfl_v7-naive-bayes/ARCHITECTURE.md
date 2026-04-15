# HFL v7 - Gaussian Naive Bayes

Esta carpeta mantiene el estilo de `hfl_v7-RN`, pero el modelo base no es una RN entrenada directamente sobre etiquetas duras.

## Esquema teacher -> student

- Teacher original: `Gaussian Naive Bayes`
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

- Teacher accuracy: 0.7467
- Student accuracy: 0.9042
- Teacher F1 macro: 0.7447
- Student F1 macro: 0.9030

## Artefactos

- `ids_3class.h5`: student model compatible con el gateway HFL.
- `ids_3class.keras`: opcional, si se activa guardado nativo de Keras desde Linux.
- `model_weights.h`: pesos base para ESP32.
- `teacher_naive_bayes.joblib`: teacher clasico original.
- `training_summary.json`: metricas y configuracion del entrenamiento.
