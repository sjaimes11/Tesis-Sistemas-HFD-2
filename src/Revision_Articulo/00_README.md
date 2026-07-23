# Revisión del artículo MDPI — entregables

Revisión de "An Edge–Fog–Cloud Architectural Framework for Distributed IoT Intrusion Detection Based on Hierarchical Federated Learning" contra la tesis `Rev1__Tesis_Sistemas_7`.

| Archivo | Contenido | Actividad |
|---|---|---|
| `01_Revision_Profunda.md` | Revisión sección por sección (contenido, coherencia, redacción, gaps) con severidad y checklist accionable | 1 |
| `02_Tabla12_y_Datos.md` | Tabla 12 llena: accuracy federada real por escenario + P/R/F1 proxy (modelos offline) + LaTeX listo + nota al pie | 2 |
| `03_Bibliografia_y_CrossRefs.md` | Parches para 5 `\bibitem` duplicados, referencias sin citar, y 2 cross-refs rotas (`sec:evaluation_methodology`, doble `fig:experimental_workflow`) | 3 y 4 |
| `figures_en/` | 5 figuras de datos regeneradas en inglés (Fig 6, 7, 9, 10, 11) desde los CSV reales / Tablas 9–10 | 3 |

## Datos clave verificados (todos reales)

**Accuracy federada por escenario** (= Tabla 4-6 tesis):
E1 93.81 · E2 92.98 · E3 96.33 · E4 90.00 · E5 96.04 · E6 93.89 · E7 97.50 · E8 94.44

**P/R/F1 proxy offline** (test set 51 255 flujos, weighted):
MLP 92.11 / 90.72 / 90.82 · CNN 92.09 / 90.70 / 90.80
(valida contra Tabla 4-3 tesis: MLP 90.60 / CNN 90.51 ✓)

## Pendiente de tu decisión / acción manual

1. **§4.8 (RQ4, overhead cripto) está en la sección de Metodología, no en Resultados** — moverla (ver `01`).
2. **Caveat Tabla 12**: accuracy federada (hasta 97.5%) vs F1 offline (~90.8%) en la misma fila — decide opción A/B/C en `02`.
3. **RQ3 (CPU/Mem)**: por tu indicación, se ignoró.
4. **Figuras conceptuales** (Fig 1–5, 8): son diagramas; si tienen texto en español hay que reeditarlas en draw.io/PowerPoint (dime si las rehago).
5. **TODOs de metodología y `% TODO-REF`**: trasladar datos de la tesis y añadir literatura 2025–2026 (puedo hacerlo).
