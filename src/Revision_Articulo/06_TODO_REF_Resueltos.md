# Resolución de los 29 `% TODO-REF`

**Principio aplicado:** ninguna referencia inventada. Las 9 referencias nuevas fueron verificadas con Crossref / PubMed Central (autores, volumen, número de artículo y DOI confirmados). Además se reutilizan 5 entradas que ya están en tu bibliografía pero **sin citar** (huérfanas), lo que resuelve de paso ese problema del doc `03`.

---

## ⚠️ HALLAZGO CRÍTICO — léelo antes que nada

Al buscar literatura para el *research gap* apareció esto:

> **Alshammari, A.** "A unified low-carbon cybersecurity framework integrating energy-efficient intrusion detection, lightweight cryptography, and carbon-aware scheduling for edge–cloud architectures." *Scientific Reports* **2026**, *16*, 10603. DOI: 10.1038/s41598-026-44260-7

Ese trabajo (**GreenShield**) integra **exactamente** tu combinación: IDS basado en deep learning + **criptografía ligera ASCON** + **aprendizaje federado jerárquico** + arquitectura **edge–fog–cloud**.

**Por qué importa:** tu §2.4 afirma que "continúa siendo limitada la evidencia experimental sobre arquitecturas que integren simultáneamente inferencia TinyML, aprendizaje federado jerárquico y protección criptográfica ligera". Un revisor que conozca GreenShield puede considerar que esa afirmación es insostenible si no lo citas.

**Tu diferenciador sigue siendo válido y hay que hacerlo explícito:** GreenShield se evalúa sobre los datasets UNSW-NB15 y CIC-IDS2017 (evaluación centrada en datos/simulación), mientras que tu trabajo **despliega y mide el ciclo completo en hardware real** (ESP32-S3 + Raspberry Pi 4 + PC), con instrumentación por canal y por ronda. Ese es el argumento a defender.

**Acción recomendada:** citarlo en §2.3, §2.4 y añadirlo como fila en la Tabla 14 (comparación de capacidades). Redacción sugerida al final de este documento.

---

## A. Referencias nuevas verificadas (9)

Pegar en el bloque `\begin{thebibliography}` con el formato MDPI que ya usas:

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

### Verificación (para tu tranquilidad)

| Clave | DOI | Verificado en |
|---|---|---|
| Lim2025 | 10.3390/electronics14203972 | Crossref |
| Tamuka2026 | 10.3390/computers15030169 | Crossref |
| Alshammari2026 | 10.1038/s41598-026-44260-7 | Crossref |
| HeydariSurvey2025 | 10.3390/s25103191 | Crossref + PMC |
| Bartoli2025 | arXiv:2505.15622 | arXiv |
| Dritsas2025 | 10.3390/jsan14010009 | Crossref |
| Cooray2025 | 10.3389/fcomp.2025.1617597 | Crossref |
| Ma2024 | 10.1186/s13677-024-00721-w | Crossref |
| Sorescu2025 | 10.3390/s25185887 | PMC |
| Fusco2025 | 10.1007/978-3-031-87769-8_34 | Crossref |
| Sun2026 | 10.3390/electronics15132879 | Crossref |

> ⚠️ `Bartoli2025` es un **preprint de arXiv**, no artículo revisado por pares. Es admisible, pero si prefieres solo peer-reviewed, usa `HeydariSurvey2025` + `Banbury2020` para ese TODO y elimínalo.

> Nota: `HeydariSurvey2025` es de **Heydari, S.**; tu `Heydari2024` existente es de **Heydari, F.** Son entradas distintas — verifica que no haya confusión de iniciales al compilar.

---

## B. Reutilización de referencias huérfanas (ya en tu .bib, sin citar)

Esto mata dos pájaros: llena TODOs y elimina las entradas sin citar que señalé en el doc `03`.

| Clave existente | Contenido | Dónde usarla ahora |
|---|---|---|
| `Banbury2020` | Benchmarking TinyML Systems | TODO 5 (métricas TinyML) |
| `Warden2019` | Libro TinyML (O'Reilly) | TODO 5 / TODO 6 |
| `Moustafa2019` | *Statistical flow features* para tráfico IoT | TODO 18 (representación compacta) |
| `Zouhri2024` | Feature selection en IDS | TODO 18 |
| `Li2017` | Feature Selection: A Data Perspective | TODO 18 |
| `CICIoT2023` | Dataset CICIoT2023 | §4.2 (dataset) |

---

## C. Mapeo TODO → cita (los 29)

Reemplaza cada comentario `% TODO-REF: ...` añadiendo el `\cite{}` al final de la frase anterior.

### Introduction

| # | TODO | Cita a insertar |
|---|---|---|
| 1 | Tendencias ciberseguridad IoT/IIoT 2025–26 | `\cite{Lim2025}` |
| 2 | Arquitecturas Edge–Fog–Cloud para ciberseguridad IoT | `\cite{Tamuka2026}` |
| 3 | Integración IA + FL + seguridad en Edge AI | `\cite{Alshammari2026,Dritsas2025}` |

### §2.1 TinyML-Based IDS

| # | TODO | Cita |
|---|---|---|
| 4 | Evolución arquitecturas Edge para IDS IoT | `\cite{Tamuka2026}` |
| 5 | Métricas de evaluación TinyML (accuracy/latency/memoria/energía) | `\cite{HeydariSurvey2025,Bartoli2025,Banbury2020}` |
| 6 | Continual learning / on-device adaptation | `\cite{HeydariSurvey2025,Warden2019}` |

### §2.2 Federated Learning

| # | TODO | Cita |
|---|---|---|
| 7 | Aprendizaje continuo / adaptación dinámica IDS IoT | `\cite{HeydariSurvey2025,Dritsas2025}` |
| 8 | Impacto de la comunicación en la convergencia FL | `\cite{Cooray2025,Ma2024}` |
| 9 | Comparativo FL vs HFL (escalabilidad, latencia, tráfico) | `\cite{Ma2024,Cooray2025}` |
| 10 | Ataques de inferencia / poisoning / privacidad de gradientes | `\cite{Dritsas2025,Cooray2025}` |

### §2.3 Lightweight Cryptography

| # | TODO | Cita |
|---|---|---|
| 11 | Confiabilidad / protección del conocimiento en FL | `\cite{Alshammari2026,Dritsas2025}` |
| 12 | Comparativo de algoritmos de criptografía ligera | `\cite{Sorescu2025}` |
| 13 | AEAD dentro de procesos FL / Edge AI | `\cite{Alshammari2026}` |
| 14 | Overhead criptográfico en FL para IoT | `\cite{Alshammari2026,Sorescu2025}` |

### §2.4 Research Gap

| # | TODO | Cita |
|---|---|---|
| 15 | Trabajos que integran parcialmente TinyML/FL/cripto | `\cite{Alshammari2026,Ramadan2025,Tran2026}` |
| 16 | Trade-offs rendimiento / comunicación / seguridad | `\cite{Alshammari2026,Cooray2025}` |

### §3 Proposed Architecture

| # | TODO | Cita |
|---|---|---|
| 17 | Survey Edge AI / TinyML para IDS IoT | `\cite{HeydariSurvey2025,Tamuka2026}` |
| 18 | Representaciones compactas de tráfico para IDS | `\cite{Fusco2025,Sun2026,Moustafa2019}` |
| 19 | Desacoplamiento inferencia Edge / FL | `\cite{Ma2024,Dritsas2025}` |
| 20 | Arquitecturas jerárquicas E–F–C para aprendizaje distribuido | `\cite{Ma2024,Cooray2025}` |
| 21 | Privacidad y eficiencia de comunicación en HFL | `\cite{Ma2024,Cooray2025}` |
| 22 | HFL y agregación multinivel entre gateways | `\cite{Ma2024,Tran2026}` |
| 23 | Coordinación global en HFL | `\cite{Ma2024,Dritsas2025}` |
| 24 | Sincronización / convergencia / coordinación global HFL | `\cite{Ma2024,Cooray2025}` |
| 25 | Survey HFL / flujos FL multinivel | `\cite{Cooray2025,Ma2024}` |
| 26 | Eficiencia de comunicación y privacidad en HFL | `\cite{Ma2024,Dritsas2025}` |
| 27 | Evaluaciones comparativas FL / HFL / cripto ligera | `\cite{Alshammari2026,Cooray2025}` |
| 28 | Seguridad en FL jerárquico / Edge AI | `\cite{Alshammari2026,Dritsas2025}` |
| 29 | Integración transparente de mecanismos criptográficos | `\cite{Alshammari2026,Sorescu2025}` |

---

## D. Texto sugerido para diferenciarte de GreenShield

### D.1 — Añadir en §2.3 (Lightweight Cryptography), tras el párrafo que empieza "La literatura reciente ha estudiado ampliamente el desempeño…"

```latex
Un caso reciente ilustra tanto el interés como los límites actuales de esta línea. Alshammari \cite{Alshammari2026} propone un marco unificado que combina detección de intrusiones eficiente en energía, criptografía ligera basada en ASCON y aprendizaje federado jerárquico sobre una organización edge--fog--cloud, cuantificando el ahorro energético y la reducción de emisiones asociadas al proceso. No obstante, su validación se apoya en los conjuntos de datos UNSW-NB15 y CIC-IDS2017 evaluados sobre infraestructura convencional, por lo que el costo real de las operaciones criptográficas y del ciclo federado sobre microcontroladores permanece sin caracterizar experimentalmente.
```

### D.2 — Ajustar el 2º párrafo de §2.4 (Research Gap)

**ANTES:**
> Aunque existen propuestas que combinan parcialmente algunos de estos componentes, la evidencia experimental sobre arquitecturas que integren simultáneamente inferencia TinyML, aprendizaje federado jerárquico y protección criptográfica ligera dentro de un mismo sistema Edge--Fog--Cloud implementado completamente sobre hardware real continúa siendo limitada.

**DESPUÉS:**
```latex
Existen ya propuestas que combinan varios de estos componentes dentro de una misma arquitectura. Alshammari \cite{Alshammari2026} integra detección de intrusiones, criptografía ligera ASCON y aprendizaje federado jerárquico en una organización edge--fog--cloud, mientras que Tran et al. \cite{Tran2026} incorporan agregación jerárquica y mecanismos criptográficos sobre plataformas embebidas, y Ramadan et al. \cite{Ramadan2025} sistematizan las prácticas actuales de aprendizaje federado con TinyML en dispositivos de borde. Sin embargo, estas propuestas se evalúan predominantemente sobre conjuntos de datos de referencia o infraestructuras convencionales, de manera que la caracterización experimental del comportamiento conjunto de las tres tecnologías --- medida extremo a extremo sobre microcontroladores reales, con instrumentación por canal de comunicación y por ronda federada --- continúa siendo limitada. Es precisamente en esa evidencia empírica, y no en la propuesta de un nuevo algoritmo, donde se sitúa la contribución de este trabajo.
```

### D.3 — Añadir fila a la Tabla 14 (§6.2), antes de "Este trabajo"

```latex
Alshammari & \checkmark & --          & \checkmark & \checkmark  & \checkmark  & --          \\
```

Es decir: IDS ✓ · TinyML–Edge ✗ · FL ✓ · HFL ✓ · Comunicaciones seguras ✓ · **Hardware–Edge real ✗**.
Esa última columna es tu diferenciador y queda visualmente evidente en la tabla.

---

## E. Nota sobre TODO 18 (resuelto con refs específicas)

Se resolvió con dos referencias nuevas verificadas, ambas de 2025–2026:

- **`Fusco2025`** — IDS TinyML con red siamesa desplegado **en MCU**. Es del mismo grupo de investigación que tu `Ficco2024` (Palmieri, Ficco), lo que refuerza la coherencia de tu marco teórico.
- **`Sun2026`** — EPC-TinyAD, framework TinyML comprimido para detección de anomalías en dispositivos de borde restringidos.

Se acompañan de `Moustafa2019` (*statistical flow features* para tráfico IoT), que ya estaba en tu bibliografía sin citar.

> Durante la búsqueda descarté un candidato porque la DOI que verifiqué correspondía en realidad a otro artículo (monitoreo de ocupación en bibliotecas). Solo quedaron las referencias cuya autoría y DOI pude confirmar.

## F. Lo que queda pendiente

- Varias referencias nuevas se reutilizan en múltiples TODOs (normal en una revisión), pero si prefieres mayor diversidad bibliográfica puedo ampliar la búsqueda por tema.
- No verifiqué si tu institución tiene acceso a todos los PDFs; los DOIs sí están confirmados.
