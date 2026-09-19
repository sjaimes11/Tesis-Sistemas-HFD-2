# paper_final/ — el .tex con todos los cambios aplicados

Carpeta `paper_final/` con `paper.tex` compilable, `Definitions/` (clase MDPI) y `Figures/`.

## Bugs de compilación corregidos

Estos rompían la compilación o producían salida incorrecta. No los había detectado antes porque solo tenía fragmentos del `.tex`.

| # | Problema | Efecto | Corrección |
|---|---|---|---|
| 1 | `The% accountaudit, acn, ...` en el preámbulo | **Error fatal** de compilación: texto suelto antes de `\begin{document}` | Eliminado (bloque de comentarios de journals removido) |
| 2 | Dos tablas con `\label{tab:detection_capability}` | Label duplicado; una tenía `TODO` en todas las celdas | Eliminada la tabla con placeholders; queda la de datos reales |
| 3 | Dos figuras con `\label{fig:experimental_workflow}` | Todos los `\ref` apuntaban a una; la otra quedaba huérfana | La del protocolo pasó a `fig:experimental_protocol` |
| 4 | `\includegraphics[width=0\textwidth]{Round_Duration_English.png}` | **Figura invisible** (ancho cero) | Cambiado a `\textwidth` |
| 5 | `\ref{sec:experimental_scenarios}` sin `\label` | Salía `Subsección ??` | Añadido `\label{sec:experimental_scenarios}` |
| 6 | `\ref{sec:results}` sin `\label` | Salía `Sección ??` | Añadido `\label{sec:results}` |
| 7 | Párrafos de §5.3 duplicados (TODO + versión escrita) | Texto repetido dos veces | Eliminadas las versiones con `TODO` |
| 8 | `...para interpretar la siguiente.limitadolmente, se verifica...` | Texto corrupto | Corregido a `...la siguiente. Inicialmente, se verifica...` |
| 9 | `los los requisitos computacionales` | Typo | Corregido |
| 10 | `RNN` en abreviaturas | Nunca se usa en el texto | Eliminado |

## Bibliografía

**6 `\bibitem` duplicados eliminados**: `Ramadan2025`, `Ficco2024` (×2 — una era *otro* paper, Smart Grids, con clave en colisión), `SaravanaBalaji2023`, `Hajj2023`, `Banbury2020`.

**13 referencias nuevas añadidas al final** (todas verificadas contra Crossref/arXiv):

| Clave | Referencia | DOI |
|---|---|---|
| Lim2025 | Electronics 2025, 14, 3972 | 10.3390/electronics14203972 |
| Tamuka2026 | Computers 2026, 15, 169 | 10.3390/computers15030169 |
| Alshammari2026 | Scientific Reports 2026, 16, 10603 | 10.1038/s41598-026-44260-7 |
| HeydariSurvey2025 | Sensors 2025, 25, 3191 | 10.3390/s25103191 |
| Bartoli2025 | arXiv:2505.15622 | arXiv |
| Dritsas2025 | JSAN 2025, 14, 9 | 10.3390/jsan14010009 |
| Cooray2025 | Front. Comput. Sci. 2025, 7, 1617597 | 10.3389/fcomp.2025.1617597 |
| Ma2024 | J. Cloud Comput. 2024, 13, 161 | 10.1186/s13677-024-00721-w |
| Sorescu2025 | Sensors 2025, 25, 5887 | 10.3390/s25185887 |
| Fusco2025 | AINA 2025, pp. 389--402 | 10.1007/978-3-031-87769-8_34 |
| Sun2026 | Electronics 2026, 15, 2879 | 10.3390/electronics15132879 |
| **Ren2024** | ACM TECS 2024, 23, 1--32 | 10.1145/3665278 |
| **Bunko2026** | AI Review 2026, 59, 125 | 10.1007/s10462-026-11519-4 |

Las dos últimas son nuevas de esta ronda y sustituyen las citas genéricas que había puesto antes:

- **Ren2024** — *On-Device Online Learning and Semantic Management of TinyML Systems*. Es específica sobre aprendizaje continuo y adaptación on-device en TinyML, que era justo el hueco de los TODO-REF de §2.1 y §2.2.
- **Bunko2026** — *A Survey of Privacy-Preserving Federated Learning for Intrusion Detection Systems*. Específica sobre ataques de inferencia, poisoning y privacidad en FL **aplicado a IDS**, que era el hueco del TODO-REF sobre ataques al modelo.

Las 29 citas `% TODO-REF` quedaron insertadas y los comentarios eliminados.

## GreenShield (Alshammari 2026)

Integrado en tres lugares, porque es el trabajo más cercano a la propuesta (IDS + ASCON + HFL + Edge--Fog--Cloud):

1. **§2.3** — párrafo nuevo que lo describe y marca su límite (validado sobre datasets, no sobre microcontroladores).
2. **§2.4** — el párrafo de *research gap* reescrito: ya no afirma que casi no existan propuestas integradas, sino que las que existen no se miden extremo a extremo sobre hardware real.
3. **Tabla 14** — fila nueva con ✗ en *Hardware--Edge real*, que es el diferenciador; además se añadieron `\cite{}` a todas las filas.

## RQ3

Se eliminaron las menciones a *CPU Utilization* y *Memory Utilization*, que no se midieron. RQ3 ahora promete latencia de inferencia y huella de memoria, que sí están en la Tabla 11. Lo mismo en §4.7.4, donde *Resource Overhead* pasó a *Communication Overhead* (el overhead de bytes sí está medido).

## Figuras

Las 5 de datos usan las versiones en inglés generadas desde los CSV reales.

Las 6 conceptuales son **placeholders rojos** — no tengo tus diagramas, viven solo en tu Overleaf. Sustitúyelos antes de enviar:
`architecture.png`, `Hierarchical Federated Learning Workflow.png`, `experimental_workflow.png`, `experimental_protocol.png`, `Model_Selection.png`, `fl_evaluation_framework.png`.

## Clase MDPI

`Definitions/` trae un `mdpi.cls` de un espejo público, para poder compilar aquí. **Tu Overleaf ya tiene el oficial**: al pasar el `.tex`, conserva tu carpeta `Definitions/` y no subas esta.
