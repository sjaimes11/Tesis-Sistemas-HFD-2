# Bibliografía y cross-references — correcciones

Como el `.tex` está en Overleaf (no en disco), aquí van parches exactos "buscar → reemplazar".

---

## A. Claves `\bibitem` DUPLICADAS (5) — causan las referencias repetidas del PDF

En un `thebibliography` manual, definir dos veces la misma clave genera "multiply-defined labels" y numeración incorrecta. **Borra estas 5 entradas** (son la 2ª aparición de cada clave):

| Clave duplicada | Qué es la 2ª aparición | Acción |
|---|---|---|
| `Ramadan2025` | Copia idéntica de "Federated Learning and TinyML…" | **Borrar** el 2º `\bibitem{Ramadan2025}` |
| `Ficco2024` | Copia idéntica de "Federated Learning for IoT Devices…" | **Borrar** ese `\bibitem{Ficco2024}` |
| `Ficco2024` | Un paper DISTINTO: "A Fog-Edge-Enabled IDS for Smart Grids, J. Cloud Computing" (colisión de clave) | **Borrar** o renombrar a `FiccoSmartGrid2024` (no está citado) |
| `SaravanaBalaji2023` | Copia idéntica de "IoT Integrated Edge Platform…" | **Borrar** el 2º |
| `Hajj2023` | Copia de "Cross-Layer Federated Learning…" con doi | **Borrar** el 2º |
| `Banbury2020` | "TinyML Systems: Challenges and Direction" (casi idéntica a "Benchmarking TinyML Systems") | **Borrar** el 2º |

Concretamente, en el bloque `\begin{thebibliography}` elimina estos `\bibitem` (los que aparecen **después** de `\bibitem{Albanbay2025}`):

```latex
% --- BORRAR: duplicado de Ramadan2025 ---
\bibitem{Ramadan2025}
Ramadan, M.N.A.; Abdo, A.R.; Eldin, A.T. Federated Learning and TinyML on IoT Edge Devices...

% --- BORRAR: duplicado de Ficco2024 ---
\bibitem{Ficco2024}
Ficco, M.; Guerriero, A.; Milite, E.; Palmieri, F.; Pietrantuono, R.; Russo, S. Federated Learning for IoT Devices...
```

Y más abajo:

```latex
% --- BORRAR: duplicado de SaravanaBalaji2023 (justo antes de Wang2019) ---
\bibitem{SaravanaBalaji2023}
Saravana Balaji, B.; ... IoT Integrated Edge Platform for Secure Industrial Application...

% --- BORRAR o RENOMBRAR: Ficco2024 = paper de Smart Grids (clave colisiona) ---
\bibitem{Ficco2024}
Ficco, M., Palmieri, F., and Castiglione, A. A Fog-Edge-Enabled Intrusion Detection System for Smart Grids...

% --- BORRAR: duplicado de Hajj2023 ---
\bibitem{Hajj2023}
Hajj, S., Azar, J., ... Cross-Layer Federated Learning for Lightweight IoT Intrusion Detection Systems. Sensors, 23(16):7038...

% --- BORRAR: duplicado de Banbury2020 ---
\bibitem{Banbury2020}
Banbury, C. R., ... TinyML Systems: Challenges and Direction. arXiv:2003.04821, 2020.
```

## B. `\bibitem` sin citar en el texto (opcional)

Estas entradas existen en la bibliografía pero **ningún `\cite{}` las usa**. MDPI pide que toda referencia listada esté citada. O las citas donde corresponda, o las eliminas:

- `CICIoT2023`, `CICFlowMeter`, `Zouhri2024`, `Li2017`, `Warden2019`, y `Banbury2020` (si borras el duplicado, la que queda también está sin citar).

Sugerencia: `CICIoT2023` y `Moustafa2019` encajan en §4.2 (dataset); `Warden2019`/`Banbury2020` en §2.1 (TinyML); `Li2017`/`Zouhri2024` en la selección de 13 features. Puedo insertar los `\cite` si me dices.

---

## C. Cross-references ROTAS

### C.1 — `\ref{sec:evaluation_methodology}` → "Sección ??"

§4.7 (Métricas de evaluación) contiene:
```latex
... definidas en la Secci\'on~\ref{sec:evaluation_methodology}.
```
pero **ningún** `\label{sec:evaluation_methodology}` existe. La subsección que define las 5 RQs es `\subsection{Metodología de evaluación}`. **Añade el label ahí:**

```latex
% ANTES
\subsection{Metodología de evaluación}
La arquitectura propuesta integra múltiples componentes...

% DESPUÉS
\subsection{Metodología de evaluación}
\label{sec:evaluation_methodology}
La arquitectura propuesta integra múltiples componentes...
```

### C.2 — `\label{fig:experimental_workflow}` DUPLICADO en dos figuras

La Figura 3 ("Experimental evaluation workflow") y la Figura 4 ("Experimental protocol workflow") usan **el mismo** `\label{fig:experimental_workflow}`. Todos los `\ref` apuntan a una sola y la otra queda huérfana.

**Renombra la segunda** (la del protocolo) y su referencia:

```latex
% En la figura del PROTOCOLO (§Protocolo experimental) — ANTES
\begin{figure*}[ht]
    \centering
    \includegraphics[width=\textwidth]{Figures/experimental_protocol.png}
    \caption{Experimental protocol workflow.}
    \label{fig:experimental_workflow}   % <-- duplicado
\end{figure*}

% DESPUÉS
\begin{figure*}[ht]
    \centering
    \includegraphics[width=\textwidth]{Figures/experimental_protocol.png}
    \caption{Experimental protocol workflow.}
    \label{fig:experimental_protocol}
\end{figure*}
```

Y en el párrafo de esa subsección:
```latex
% ANTES
La Figura~\ref{fig:experimental_workflow} resume las tres fases que componen cada ronda del protocolo.
% DESPUÉS
La Figura~\ref{fig:experimental_protocol} resume las tres fases que componen cada ronda del protocolo.
```

> Antes de esto: **confirma que Figura 3 y Figura 4 son diagramas distintos**. Si son el mismo diagrama, deja una sola figura y un solo label.

### C.3 — Verificar orden de figuras

Con §4.8 movida a Resultados (ver `01_Revision_Profunda`), revisa que la numeración de figuras siga el orden de aparición (Figs 6 y 7 quedarán después de la 11 si mueves §4.8 al final de §5). LaTeX renumera solo, pero cuida que el texto no diga "Figura 6" esperando que aparezca antes que la 9.

---

## D. Figuras en inglés (Actividad 3)

Regeneradas desde los CSV reales en: `Revision_Articulo/figures_en/`

| Archivo nuevo | Reemplaza en el `.tex` | Figura del artículo |
|---|---|---|
| `fig6_crypto_latency.png` | `Figures/Crypto_Latency.png` | Fig 6 — p95 latency |
| `fig7_payload_overhead.png` | `Figures/Payload_Overhead.png` | Fig 7 — payload/overhead |
| `fig9_convergence.png` | `Figures/convergence.png` | Fig 9 — accuracy/loss |
| `fig10_weight_stability.png` | `Figures/stability.png` | Fig 10 — pesos federados |
| `fig11_round_duration.png` | `Figures/Round_Duration.png` | Fig 11 — duración/ronda |

Leyendas ya en inglés con etiquetas E1–E8 (MLP/CNN · Edge-Cloud/Edge-Fog-Cloud · ASCON/PLAIN), consistentes con la Tabla 6.

**Figuras conceptuales** (Fig 1 arquitectura, Fig 2 HFL workflow, Fig 3/4 workflows, Fig 5 y Fig 8 frameworks): son diagramas, no gráficas de datos. Si sus textos internos están en español, hay que reeditarlas en la herramienta original (draw.io/PowerPoint). Dime si quieres que las rehaga.
