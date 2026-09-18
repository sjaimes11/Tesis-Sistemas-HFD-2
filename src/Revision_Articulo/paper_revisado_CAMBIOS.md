# Cambios aplicados — guía de aplicación en Overleaf

El `.tex` vive en Overleaf, así que entrego los cambios como **parches ordenados por posición en el documento**. Cada bloque es "buscar → reemplazar" o "insertar". Aplícalos de arriba hacia abajo.

Total: **10 grupos de cambios**. Tiempo estimado: 25–35 min.

---

# GRUPO 1 — Bibliografía: borrar duplicados

En `\begin{thebibliography}`, **borra** estos 6 `\bibitem` (son segundas apariciones de claves ya definidas). Están todos **después** de `\bibitem{Albanbay2025}`:

1. El 2º `\bibitem{Ramadan2025}` (justo después de Albanbay2025)
2. El 2º `\bibitem{Ficco2024}` (el que sigue al anterior)
3. El 2º `\bibitem{SaravanaBalaji2023}` (antes de `Wang2019`)
4. El `\bibitem{Ficco2024}` de *"A Fog-Edge-Enabled Intrusion Detection System for Smart Grids"* ← **colisión de clave: es otro paper**
5. El 2º `\bibitem{Hajj2023}` (el que trae `doi:10.3390/s23167038`)
6. El 2º `\bibitem{Banbury2020}` (*"TinyML Systems: Challenges and Direction"*, el último de la lista)

---

# GRUPO 2 — Bibliografía: añadir 11 referencias nuevas

Pega esto **antes** de `\end{thebibliography}`:

```latex
\bibitem{Lim2025}
Lim, K.S.; Ooi, S.Y.; Sayeed, M.S.; Chew, Y.J.; Ahmad, N.M. Securing the Internet of Things: Systematic Insights into Architectures, Threats, and Defenses. \emph{Electronics} \textbf{2025}, \emph{14}, 3972.

\bibitem{Tamuka2026}
Tamuka, N.; Mathonsi, T.E.; Olwal, T.O.; Maswikaneng, S.; Muchenje, T.; Tshilongamulenzhe, T.M. Intrusion Detection in Fog Computing: A Systematic Review of Security Advances and Challenges. \emph{Computers} \textbf{2026}, \emph{15}, 169.

\bibitem{Alshammari2026}
Alshammari, A. A Unified Low-Carbon Cybersecurity Framework Integrating Energy-Efficient Intrusion Detection, Lightweight Cryptography, and Carbon-Aware Scheduling for Edge--Cloud Architectures. \emph{Scientific Reports} \textbf{2026}, \emph{16}, 10603.

\bibitem{HeydariSurvey2025}
Heydari, S.; Mahmoud, Q.H. Tiny Machine Learning and On-Device Inference: A Survey of Applications, Challenges, and Future Directions. \emph{Sensors} \textbf{2025}, \emph{25}, 3191.

\bibitem{Bartoli2025}
Bartoli, P.; Veronesi, C.; Giudici, A.; Siorpaes, D.; Trojaniello, D.; Zappa, F. Benchmarking Energy and Latency in TinyML: A Novel Method for Resource-Constrained AI. \emph{arXiv} \textbf{2025}, arXiv:2505.15622.

\bibitem{Dritsas2025}
Dritsas, E.; Trigka, M. Federated Learning for IoT: A Survey of Techniques, Challenges, and Applications. \emph{Journal of Sensor and Actuator Networks} \textbf{2025}, \emph{14}, 9.

\bibitem{Cooray2025}
Cooray, L.; Sendanayake, J.; Vithanaarachchi, P.; Priyadarshana, Y.H.P.P. Deep Federated Learning: A Systematic Review of Methods, Applications, and Challenges. \emph{Frontiers in Computer Science} \textbf{2025}, \emph{7}, 1617597.

\bibitem{Ma2024}
Ma, C.; Li, X.; Huang, B.; Li, G.; Li, F. Personalized Client-Edge-Cloud Hierarchical Federated Learning in Mobile Edge Computing. \emph{Journal of Cloud Computing} \textbf{2024}, \emph{13}, 161.

\bibitem{Sorescu2025}
Sorescu, T.-G.; Chiriac, V.-M.; Stoica, M.-A.; Comsa, C.-R.; Soroaga, I.-G.; Contac, A. Comparative Performance Analysis of Lightweight Cryptographic Algorithms on Resource-Constrained IoT Platforms. \emph{Sensors} \textbf{2025}, \emph{25}, 5887.

\bibitem{Fusco2025}
Fusco, P.; Montefusco, A.; Rimoli, G.P.; Palmieri, F.; Ficco, M. TinyML-Based Intrusion Detection System for Handling Class Imbalance in IoT-Edge Domain Using Siamese Neural Network on MCU. In \emph{Advanced Information Networking and Applications (AINA 2025)}; Lecture Notes on Data Engineering and Communications Technologies; Springer: Cham, Switzerland, 2025; pp.~389--402.

\bibitem{Sun2026}
Sun, Y.; Qin, Y.; Chen, W.; Zhao, W.; Sun, H. EPC-TinyAD: An Energy- and Privacy-Aware Compressed TinyML Framework for Reliable Industrial Anomaly Detection on Resource-Constrained Edge Devices. \emph{Electronics} \textbf{2026}, \emph{15}, 2879.
```

**Todas verificadas** (autores + DOI vía Crossref/PMC). Tabla de verificación en `06_TODO_REF_Resueltos.md`.

---

# GRUPO 3 — Cross-references rotas

### 3.1 Añadir label faltante

```latex
% BUSCAR
\subsection{Metodología de evaluación}
La arquitectura propuesta integra múltiples componentes tecnológicos,

% REEMPLAZAR
\subsection{Metodología de evaluación}
\label{sec:evaluation_methodology}
La arquitectura propuesta integra múltiples componentes tecnológicos,
```

### 3.2 Label duplicado de figura

En la figura del **protocolo experimental** (`Figures/experimental_protocol.png`):

```latex
% BUSCAR
    \caption{Experimental protocol workflow.}
    \label{fig:experimental_workflow}

% REEMPLAZAR
    \caption{Experimental protocol workflow.}
    \label{fig:experimental_protocol}
```

Y en el párrafo de esa subsección:

```latex
% BUSCAR
La Figura~\ref{fig:experimental_workflow} resume las tres fases que componen cada ronda del protocolo.
% REEMPLAZAR
La Figura~\ref{fig:experimental_protocol} resume las tres fases que componen cada ronda del protocolo.
```

---

# GRUPO 4 — Las 29 citas `% TODO-REF`

Añade el `\cite{}` al final de la frase **anterior** a cada comentario, y borra el comentario.

## Introduction
| TODO | Frase que termina en… | Cita |
|---|---|---|
| 1 | "…ataques cada vez más sofisticados \cite{Ni2024,Amanullah2020,Bagaa2020}" | añadir `Lim2025` → `\cite{Ni2024,Amanullah2020,Bagaa2020,Lim2025}` |
| 2 | "…cada nivel de la infraestructura \cite{SaravanaBalaji2023,Ni2024,Dini2024}" | → `\cite{SaravanaBalaji2023,Ni2024,Dini2024,Tamuka2026}` |
| 3 | "…dentro de una misma infraestructura." | → "…dentro de una misma infraestructura \cite{Alshammari2026,Dritsas2025}." |

## §2.1 TinyML
| TODO | Cita |
|---|---|
| 4 | `\cite{...,Tamuka2026}` (añadir a la cita existente del 1er párrafo) |
| 5 | `\cite{...,HeydariSurvey2025,Bartoli2025,Banbury2020}` |
| 6 | `\cite{Ficco2024,Ramadan2025,HeydariSurvey2025,Warden2019}` |

## §2.2 Federated Learning
| TODO | Cita |
|---|---|
| 7 | `\cite{McMahan2017,Imteaj2022,Ramadan2025,HeydariSurvey2025}` |
| 8 | `\cite{...,Cooray2025,Ma2024}` |
| 9 | `\cite{Ficco2024,Tran2026,Ramadan2025,Ma2024,Cooray2025}` |
| 10 | `\cite{Imteaj2022,Hajj2023,Dritsas2025,Cooray2025}` |

## §2.3 Lightweight Cryptography
| TODO | Cita |
|---|---|
| 11 | `\cite{Alshammari2026,Dritsas2025}` |
| 12 | `\cite{...,Sorescu2025}` |
| 13 | `\cite{...,Alshammari2026}` |
| 14 | `\cite{Alshammari2026,Sorescu2025}` |

## §2.4 Research Gap
| TODO | Cita |
|---|---|
| 15 | (ver GRUPO 5 — se reescribe el párrafo) |
| 16 | `\cite{Alshammari2026,Cooray2025}` |

## §3 Proposed Architecture
| TODO | Cita |
|---|---|
| 17 | `\cite{SaravanaBalaji2023,Ficco2024,HeydariSurvey2025,Tamuka2026}` |
| 18 | `\cite{Fusco2025,Sun2026,Moustafa2019}` |
| 19 | `\cite{Ma2024,Dritsas2025}` |
| 20 | `\cite{Ma2024,Cooray2025}` |
| 21 | `\cite{Ma2024,Cooray2025}` |
| 22 | `\cite{Ma2024,Tran2026}` |
| 23 | `\cite{Ma2024,Dritsas2025}` |
| 24 | `\cite{Ma2024,Cooray2025}` |
| 25 | `\cite{Cooray2025,Ma2024}` |
| 26 | `\cite{Ma2024,Dritsas2025}` |
| 27 | `\cite{Alshammari2026,Cooray2025}` |
| 28 | `\cite{Alshammari2026,Dritsas2025}` |
| 29 | `\cite{Alshammari2026,Sorescu2025}` |

---

# GRUPO 5 — Diferenciación de GreenShield (crítico)

### 5.1 Nuevo párrafo en §2.3

Insertar **después** del párrafo que empieza *"La literatura reciente ha estudiado ampliamente el desempeño de algoritmos de criptografía ligera…"*:

```latex
Un caso reciente ilustra tanto el interés como los límites actuales de esta línea. Alshammari \cite{Alshammari2026} propone un marco unificado que combina detección de intrusiones eficiente en energía, criptografía ligera basada en ASCON y aprendizaje federado jerárquico sobre una organización edge--fog--cloud, cuantificando el ahorro energético y la reducción de emisiones asociadas al proceso. No obstante, su validación se apoya en los conjuntos de datos UNSW-NB15 y CIC-IDS2017 evaluados sobre infraestructura convencional, por lo que el costo real de las operaciones criptográficas y del ciclo federado sobre microcontroladores permanece sin caracterizar experimentalmente.
```

### 5.2 Reescribir el 2º párrafo de §2.4

**BUSCAR** el párrafo que empieza *"A pesar de estos avances, las tres líneas de investigación han evolucionado principalmente de forma independiente…"* y termina *"…implementado completamente sobre hardware real continúa siendo limitada."*

**REEMPLAZAR su frase final** (desde "Aunque existen propuestas que combinan parcialmente…") por:

```latex
Existen ya propuestas que combinan varios de estos componentes dentro de una misma arquitectura. Alshammari \cite{Alshammari2026} integra detección de intrusiones, criptografía ligera ASCON y aprendizaje federado jerárquico en una organización edge--fog--cloud; Tran et al. \cite{Tran2026} incorporan agregación jerárquica y mecanismos criptográficos sobre plataformas embebidas; y Ramadan et al. \cite{Ramadan2025} sistematizan las prácticas actuales de aprendizaje federado con TinyML en dispositivos de borde. Sin embargo, estas propuestas se evalúan predominantemente sobre conjuntos de datos de referencia o infraestructuras convencionales, de manera que la caracterización experimental del comportamiento conjunto de las tres tecnologías --- medida extremo a extremo sobre microcontroladores reales, con instrumentación por canal de comunicación y por ronda federada --- continúa siendo limitada. Es precisamente en esa evidencia empírica, y no en la propuesta de un nuevo algoritmo, donde se sitúa la contribución de este trabajo.
```

### 5.3 Nueva fila en Tabla 14 (§6.2)

Insertar antes de `\midrule` + `\textbf{Este trabajo}`:

```latex
Alshammari             & \checkmark & --          & \checkmark & \checkmark  & \checkmark  & --          \\
```

> La columna **Hardware–Edge real** en ✗ para Alshammari es tu diferenciador, y queda visualmente evidente.

---

# GRUPO 6 — Tabla 12 (RQ1)

Reemplazar el `\begin{table*}…\end{table*}` completo de `tab:detection_capability` por la versión de `02_Tabla12_y_Datos.md` (accuracy federada real + P/R/F1 offline + nota al pie).

---

# GRUPO 7 — §5.3: los dos TODO de redacción

Reemplazar los dos `**TODO:**` por los párrafos de `04_TODOs_Seccion5_3.md`.

---

# GRUPO 8 — §6.1 y §7 alineados

Aplicar los dos parches de `05_Ajustes_Discusion_Conclusiones.md`.

---

# GRUPO 9 — Mover §4.8 a Resultados

La subsección *"Impacto de la protección criptográfica sobre el aprendizaje federado"* (con Tablas 9, 10 y Figuras 6, 7) responde **RQ4** pero está dentro de la Sección 4 (Metodología).

**Acción:** córtala completa y pégala al final de la **Sección 5 (Results)**, después de §5.4, renombrándola:

```latex
\subsection{Impacto de la protección criptográfica sobre el intercambio de parámetros}
```

Y corrige su primera frase, que actualmente dice *"La subsección anterior demostró que el proceso de aprendizaje federado mantiene un costo operativo estable…"* — ahora sí es correcta, porque §5.4 (eficiencia operacional) queda justo antes.

---

# GRUPO 10 — Limpieza menor

| Cambio | Dónde |
|---|---|
| `deteccion` → `detección` (sin tilde, ~6 veces) | §2, §6, §7 |
| Borrar `RNN & Recurrent Neural Network\\` | `\abbreviations` (no se usa RNN) |
| Unificar `NoFog`/`Fog` → `Edge--Cloud`/`Edge--Fog--Cloud` | Tablas 9, 10, 13 |
| Unificar `RN` → `MLP` | Tablas 9, 10, 13 |

---

# Figuras en inglés

Sube los 5 archivos de `Revision_Articulo/figures_en/` a la carpeta `Figures/` de Overleaf, **renombrándolos** para no tocar el `.tex`:

| Archivo generado | Renombrar a |
|---|---|
| `fig6_crypto_latency.png` | `Crypto_Latency.png` |
| `fig7_payload_overhead.png` | `Payload_Overhead.png` |
| `fig9_convergence.png` | `convergence.png` |
| `fig10_weight_stability.png` | `stability.png` |
| `fig11_round_duration.png` | `Round_Duration.png` |

Las conceptuales (Fig 1–5, 8) son diagramas: si tienen texto en español hay que reeditarlas en su herramienta original.

---

# Checklist final

- [ ] G1 — 6 duplicados borrados
- [ ] G2 — 11 referencias añadidas
- [ ] G3 — 2 cross-refs arregladas
- [ ] G4 — 29 TODO-REF citados
- [ ] G5 — GreenShield diferenciado (§2.3, §2.4, Tabla 14)
- [ ] G6 — Tabla 12 llena
- [ ] G7 — §5.3 redactada
- [ ] G8 — §6.1 y §7 alineados
- [ ] G9 — §4.8 movida a §5.5
- [ ] G10 — limpieza menor
- [ ] Figuras en inglés subidas
- [ ] Compilar y revisar que no queden `??`
