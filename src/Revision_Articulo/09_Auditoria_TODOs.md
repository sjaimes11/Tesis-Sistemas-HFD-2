# Auditoría: ¿la cita que puse responde a lo que el TODO pedía?

Reconstruí el texto literal de los 29 `% TODO-REF` desde el `.tex` original (guardado en
`paper_ORIGINAL_del_profesor.tex`) y lo comparé, uno por uno, con lo que quedó en
`paper_final/paper.tex`.

**Veredictos:**

- ✅ **CUMPLE** — la referencia es del tipo de fuente pedido y trata el tema pedido.
- 🟡 **RAZONABLE** — sostiene la frase, pero no es exactamente el tipo de fuente que se pidió.
- 🟠 **PARCIAL** — falta una parte explícita del pedido (el dominio, el tipo de fuente o el rango de años).
- 🔴 **NO CUMPLE** — la referencia no cubre lo que el TODO pedía.

**Resultado global: 14 ✅ · 5 🟡 · 9 🟠 · 1 🔴**

---

## Introduction

### TODO 1 — ✅ CUMPLE
> *"Incorporar una referencia reciente (2025–2026) que analice las tendencias actuales de ciberseguridad en IoT o IIoT."*

**L62** · añadí `Lim2025` a `\cite{Ni2024,Amanullah2020,Bagaa2020,Lim2025}`
`Lim2025` = *Securing the Internet of Things: Systematic Insights into Architectures, Threats, and Defenses*, Electronics 2025.
Es de 2025, es IoT, y es un panorama de amenazas y defensas. Encaje directo.

### TODO 2 — 🟠 PARCIAL
> *"Buscar un trabajo reciente que revise **específicamente las arquitecturas Edge–Fog–Cloud** para aplicaciones de ciberseguridad en IoT."*

**L64** · añadí `Tamuka2026`
`Tamuka2026` = *Intrusion Detection in **Fog Computing**: A Systematic Review*, Computers 2026.
**Lo que falta:** revisa el nivel **Fog** y la seguridad, pero no la arquitectura de **tres niveles** Edge–Fog–Cloud como tal. Es lo más cercano que encontré verificable; el pedido era más estrecho.

### TODO 3 — ✅ CUMPLE
> *"Incorporar una referencia reciente que aborde la integración de IA, el aprendizaje federado y la seguridad en arquitecturas de Edge AI para IoT."*

**L68** · cita nueva completa `\cite{Alshammari2026,Dritsas2025}`
`Alshammari2026` integra literalmente los tres elementos: IDS con deep learning (IA) + aprendizaje federado jerárquico + criptografía ASCON, sobre edge–fog–cloud. Es el mejor encaje de todo el lote.

---

## §2.1 TinyML-Based Intrusion Detection

### TODO 4 — 🟠 PARCIAL
> *"Incluir una referencia reciente (2025–2026) que analice específicamente **la evolución de las arquitecturas Edge** para IDS en IoT."*

**L81** · añadí `Tamuka2026`
**Lo que falta:** el foco de Tamuka es Fog, no "la evolución de las arquitecturas Edge". Además es la **misma referencia del TODO 2**, reutilizada.

### TODO 5 — ✅ CUMPLE (el mejor encaje del lote)
> *"Buscar un trabajo de revisión reciente que aborde explícitamente las métricas de evaluación para TinyML (accuracy, latency, memory footprint y energy consumption)."*

**L85** · añadí `HeydariSurvey2025,Bartoli2025` junto a `Banbury2020`
- `Bartoli2025` = *Benchmarking **Energy and Latency** in TinyML* → dos de las cuatro métricas, literal.
- `Banbury2020` = *Benchmarking TinyML Systems* → las cuatro.
- `HeydariSurvey2025` = survey de TinyML e inferencia on-device.

Es revisión, es reciente, y las métricas están nombradas explícitamente en las fuentes.

### TODO 6 — ✅ CUMPLE
> *"Incorporar una referencia reciente que analice específicamente el **aprendizaje continuo** (continual learning u on-device adaptation) aplicado a TinyML en IoT."*

**L87** · `\cite{Ficco2024,Ramadan2025,Ren2024}`
`Ren2024` = ***On-Device Online Learning** and Semantic Management of **TinyML Systems***, ACM TECS.
On-device online learning es exactamente continual learning on-device, y el dominio es TinyML. Este es el que corregí en la última ronda: antes había puesto surveys genéricos.

---

## §2.2 Federated Learning for Distributed IDS

### TODO 7 — 🟠 PARCIAL
> *"Incorporar una referencia reciente (2025–2026) que aborde específicamente la necesidad de aprendizaje continuo o de adaptación dinámica **para IDS en IoT**."*

**L92** · `\cite{McMahan2017,Imteaj2022,Ramadan2025,Ren2024}`
**Lo que falta:** `Ren2024` cubre el aprendizaje continuo on-device, pero **no el vínculo con IDS**, y es de 2024 (el pedido decía 2025–2026). Es la misma referencia del TODO 6, reutilizada.

### TODO 8 — 🟠 PARCIAL
> *"Incorporar un trabajo reciente que analice **experimentalmente** el impacto de la comunicación en la convergencia del aprendizaje federado en redes IoT."*

**L96** · añadí `Cooray2025,Ma2024`
- `Ma2024` sí es experimental y analiza comunicación y convergencia en HFL — pero en *Mobile Edge Computing*, no en redes IoT estrictamente.
- `Cooray2025` es una **revisión sistemática**, no un estudio experimental.

**Lo que falta:** el pedido decía "experimentalmente" y la mitad de la cita es un survey.

### TODO 9 — 🟡 RAZONABLE
> *"Buscar un estudio comparativo entre arquitecturas FL y HFL que **cuantifique** los beneficios de la escalabilidad, la latencia y la reducción del tráfico."*

**L98** · añadí `Ma2024,Cooray2025`
`Ma2024` compara una organización cliente–edge–cloud jerárquica contra alternativas y cuantifica resultados. Encaja. `Cooray2025` es survey general y no aporta la comparación cuantitativa pedida.

### TODO 10 — ✅ CUMPLE
> *"Incorporar una referencia reciente sobre **ataques de inferencia, poisoning de modelos o privacidad de los gradientes** en el aprendizaje federado aplicado a IoT."*

**L100** · `\cite{Imteaj2022,Hajj2023,Bunko2026}`
`Bunko2026` = *A Survey of **Privacy-Preserving Federated Learning for Intrusion Detection Systems***, AI Review 2026.
Cubre los tres vectores nombrados, en FL, y además aplicado a IDS. Corregido en la última ronda (antes tenía surveys genéricos de FL).

---

## §2.3 Lightweight Cryptography

### TODO 11 — ✅ CUMPLE
> *"Incorporar referencias recientes sobre la **confiabilidad y la protección del conocimiento** en el aprendizaje federado para IoT."*

**L105** · `\cite{Bunko2026,Alshammari2026}`
`Bunko2026` cubre la preservación de privacidad del modelo; `Alshammari2026` la protección criptográfica del ciclo federado. Las dos caras de "proteger el conocimiento".

### TODO 12 — ✅ CUMPLE (encaje literal)
> *"Incorporar un trabajo **comparativo** reciente sobre **algoritmos de criptografía ligera** para **plataformas IoT**."*

**L107** · añadí `Sorescu2025`
`Sorescu2025` = ***Comparative Performance Analysis** of **Lightweight Cryptographic Algorithms** on **Resource-Constrained IoT Platforms***, Sensors 2025.
El título responde palabra por palabra.

### TODO 13 — ✅ CUMPLE (con una salvedad)
> *"Buscar **estudios** que evalúen mecanismos de **AEAD** en procesos de aprendizaje federado o de Edge AI."*

**L113** · **párrafo nuevo** que cita `Alshammari2026`
Alshammari usa ASCON (que es AEAD) dentro de un ciclo de aprendizaje federado jerárquico. Es exactamente el escenario pedido.
**Salvedad:** el TODO decía "estudios" en plural y sólo encontré uno verificable.

**Por qué lo puse en un párrafo nuevo y no al final del párrafo anterior (L111):** esa frase afirma que la mayoría de trabajos *no* analiza el cifrado dentro del aprendizaje distribuido. Citar ahí a Alshammari, que sí lo hace, la contradiría.

### TODO 14 — 🔴 NO CUMPLE — **el más flojo, revísalo**
> *"Incorporar trabajos recientes que cuantifiquen el **overhead criptográfico** en **sistemas de aprendizaje federado** para IoT."*

**L115** · puse sólo `\cite{Sorescu2025}` dentro de la frase, tras "el costo temporal del cifrado"
`Sorescu2025` cuantifica el costo de la criptografía ligera en IoT, pero **sobre algoritmos aislados, no dentro de un sistema de aprendizaje federado**. Falta justo la mitad del pedido.

**Arreglo propuesto:** añadir `Alshammari2026`, que sí mide el overhead dentro del ciclo federado:
```latex
Aspectos como el costo temporal del cifrado \cite{Sorescu2025,Alshammari2026}, el incremento del tráfico...
```
En mi documento `06_` lo había mapeado así; al aplicarlo al `.tex` se me quedó sólo `Sorescu2025`. Es una discrepancia mía, no una decisión.

---

## §2.4 Research Gap

### TODO 15 — ✅ CUMPLE
> *"Incorporar trabajos recientes que integren parcialmente TinyML, aprendizaje federado o criptografía ligera para **delimitar con mayor precisión el alcance de la brecha**."*

**L120** · **párrafo reescrito entero** con `Alshammari2026`, `Tran2026`, `Ramadan2025`
Es lo que el TODO pedía, y además resolvió el riesgo de GreenShield: el párrafo ya no afirma que casi no existan propuestas integradas, sino que las que existen no se miden extremo a extremo sobre hardware real.

### TODO 16 — 🟡 RAZONABLE
> *"Incorporar un estudio reciente que discuta **explícitamente los trade-offs** entre rendimiento, comunicación y seguridad."*

**L122** · `\cite{Alshammari2026,Cooray2025}`
Alshammari discute compromisos entre energía, seguridad y rendimiento; Cooray recoge retos de comunicación en FL. Ninguno es un estudio **dedicado** a trade-offs, que es lo que decía "explícitamente".

---

## §3 Proposed Architecture (TODO 17–29)

### TODO 17 — 🟠 PARCIAL
> *"Incorporar un **survey** reciente (2024–2026) sobre Edge AI o TinyML aplicado a **detección de intrusiones** en IoT."*

**L168** · `\cite{SaravanaBalaji2023,Ficco2024,HeydariSurvey2025,Tamuka2026}`
`HeydariSurvey2025` es survey de TinyML pero **no de IDS**; `Tamuka2026` es review de IDS pero **no de TinyML**.
**Lo que falta:** un solo survey que cruce ambas cosas. Cubrí el tema con dos fuentes complementarias en lugar de la que se pedía.

### TODO 18 — ✅ CUMPLE
> *"Buscar trabajos recientes sobre **representaciones compactas del tráfico** para Edge AI o TinyML aplicadas a IDS."*

**L170** · añadí `Fusco2025,Sun2026,Moustafa2019`
- `Moustafa2019` = *statistical flow features* para tráfico IoT → es literalmente la representación compacta que usa tu sistema (13 características estadísticas por flujo).
- `Fusco2025` = IDS TinyML desplegado **en MCU**, del mismo grupo que tu `Ficco2024`.
- `Sun2026` = TinyML **comprimido** para detección de anomalías en el borde.

Éste es el TODO que me pediste seguir buscando, y quedó respaldado por tres ángulos.

### TODO 19 — 🟠 PARCIAL
> *"Incorporar una referencia reciente sobre el **desacoplamiento** entre la inferencia Edge y el aprendizaje federado."*

**L172** · `\cite{Ma2024,Dritsas2025}`
**Lo que falta:** ninguno trata el "desacoplamiento" como tema explícito. En `Ma2024` la separación entre inferencia local y entrenamiento jerárquico está presente, pero es una lectura mía, no la tesis del paper.

### TODO 20 — 🟠 PARCIAL
> *"Incorporar un **survey** reciente (2024–2026) sobre **arquitecturas jerárquicas Edge–Fog–Cloud** para aprendizaje distribuido."*

**L177** · `\cite{Ma2024,Cooray2025}`
**Lo que falta:** `Ma2024` es jerárquico pero **no es survey** (es una propuesta con experimentos); `Cooray2025` sí es revisión sistemática pero **no de jerarquías**. Tengo las dos mitades en referencias distintas.

### TODO 21 — 🟡 RAZONABLE
> *"Buscar una referencia reciente sobre la **privacidad** y la **eficiencia de la comunicación** en el aprendizaje federado jerárquico."*

**L181** · `\cite{Hajj2023,Chaurasia2024,Ficco2024,Ma2024,Cooray2025}`
`Ma2024` cubre bien la eficiencia de comunicación en HFL. **La parte de privacidad queda floja** — `Bunko2026` encajaría mejor aquí.

### TODO 22 — ✅ CUMPLE
> *"Incorporar referencias recientes sobre HFL y sobre la **agregación multinivel entre gateways**."*

**L183** · `\cite{Ma2024,Tran2026}`
`Ma2024` es agregación cliente–edge–cloud (multinivel); `Tran2026` es agregación jerárquica sobre plataformas embebidas (gateways). Los dos lados del pedido.

### TODO 23 — 🟠 PARCIAL
> *"Incorporar un **survey** reciente (2024–2026) sobre **coordinación global** en HFL."*

**L188** · `\cite{McMahan2017,Imteaj2022,Ramadan2025,Ma2024,Dritsas2025}`
**Lo que falta:** ninguna es un survey de coordinación global en HFL. `McMahan2017` es el paper de FedAvg, de 2017 — sostiene la frase pero está lejos del pedido.

### TODO 24 — 🟡 RAZONABLE
> *"Incorporar referencias recientes sobre **sincronización, convergencia y coordinación global** en HFL."*

**L192** · `\cite{Hajj2023,Ficco2024,Ma2024,Cooray2025}`
`Ma2024` trata convergencia en HFL con resultados propios. Cubre la parte principal; sincronización y coordinación quedan menos respaldadas.

### TODO 25 — 🟠 PARCIAL
> *"Incorporar un **survey** reciente (2024–2026) sobre **HFL** o flujos de aprendizaje federado multinivel."*

**L200** · `\cite{McMahan2017,Imteaj2022,Ramadan2025,Cooray2025}`
`Cooray2025` es survey reciente y sistemático, pero de **Deep FL en general**, no de HFL ni de flujos multinivel.

### TODO 26 — 🟡 RAZONABLE
> *"Buscar referencias recientes sobre la **eficiencia de la comunicación y la privacidad** en HFL."*

**L211** · `\cite{Hajj2023,Ficco2024,Ma2024,Dritsas2025}`
Mismo caso que el TODO 21: comunicación bien cubierta por `Ma2024`, privacidad floja.
**Nota:** los TODO 21 y 26 piden casi lo mismo y recibieron casi la misma cita.

### TODO 27 — ✅ CUMPLE
> *"Incorporar referencias recientes sobre **evaluaciones comparativas entre FL, HFL y esquemas protegidos mediante criptografía ligera**."*

**L215** · `\cite{Alshammari2026,Cooray2025}`
`Alshammari2026` compara exactamente esas tres cosas dentro de un mismo marco. Encaje directo.

### TODO 28 — ✅ CUMPLE
> *"Incorporar un **survey** reciente (2024–2026) sobre **seguridad** en arquitecturas de aprendizaje federado jerárquico o Edge AI."*

**L220** · `\cite{Alshammari2026,Bunko2026}`
`Bunko2026` es survey (2026), es de seguridad, y es de FL. **Es el único de los cinco TODO que pedían "survey" que quedó realmente satisfecho.** Corregido en la última ronda.

### TODO 29 — ✅ CUMPLE
> *"Incorporar una referencia reciente sobre la **integración transparente de mecanismos criptográficos** en arquitecturas de Edge AI o FL."*

**L224** · `\cite{NIST800232,Kaur2025,Alshammari2026,Sorescu2025}`
`Alshammari2026` integra ASCON dentro del ciclo federado sin rediseñarlo — que es justo lo que significa "integración transparente".

---

## Los tres patrones débiles

### 1. Los TODO que pedían "survey" (17, 20, 23, 25, 28)
El profesor pidió explícitamente **un survey reciente (2024–2026)** en cinco puntos. **Sólo el 28 quedó cubierto** con un survey real (`Bunko2026`). En los otros cuatro puse papers de investigación o surveys de un tema adyacente.

Los huecos concretos, si quieres que siga buscando:
- **17** — survey de *TinyML aplicado a IDS* (no TinyML por un lado e IDS por otro)
- **20** — survey de *arquitecturas jerárquicas Edge–Fog–Cloud*
- **23 / 25** — survey de *Hierarchical Federated Learning* como tal

### 2. Referencias reutilizadas en TODOs distintos
`Tamuka2026` → TODOs 2 y 4 · `Ren2024` → TODOs 6 y 7 · `Ma2024` → nueve TODOs distintos.
Es normal en una revisión, pero `Ma2024` está haciendo demasiado trabajo: sostiene casi toda la argumentación sobre HFL.

### 3. Discrepancia mía en el TODO 14
En `06_TODO_REF_Resueltos.md` mapeé `Alshammari2026,Sorescu2025`; al aplicarlo quedó sólo `Sorescu2025`. Ver el arreglo propuesto arriba.

---

## Para revisar tú mismo

- `paper_ORIGINAL_del_profesor.tex` — el `.tex` tal como lo pegaste, con los 29 `% TODO-REF` intactos
- `09_diff_original_vs_final.diff` — diff línea a línea (31 bloques de cambio)

---

# Anexo — Los 2 TODO de prosa (§5.3)

Estos no pedían citas sino escribir el análisis. Texto literal recuperado del original.

## TODO de prosa A — ✅ CUMPLE
> *"**TODO:** Incorporar una síntesis de los principales resultados observados para cada uno de estos factores, destacando únicamente las diferencias relevantes identificadas experimentalmente."*

Original L1021 → **FINAL L822**

Escribí la síntesis de los tres factores con las diferencias calculadas desde la accuracy federada real:
modelo (CNN 95.5% vs MLP 93.3%), topología (efecto no uniforme: +2.5 / +1.5 / +0.6 / −3.0) y
protección (ASCON +0.8 a +6.3, nunca negativo). Sólo diferencias medidas, como pedía.

## TODO de prosa B — ⚠️ CUMPLÍ A MEDIAS **A PROPÓSITO** — decide tú

> *"**TODO:** Incorporar el rango de variación observado en las métricas de detección entre las ocho configuraciones experimentales y **demostrar que dichas variaciones permanecen dentro de límites reducidos**. Esta evidencia permitirá concluir que la capacidad de detección constituye una propiedad robusta de la arquitectura propuesta."*

Original L1025 → **FINAL L824**

**Hice la primera mitad y no la segunda.**

- ✅ Incorporé el rango: 90.0% (E4) a 97.5% (E7).
- ❌ **No demostré que las variaciones sean "reducidas"**, porque **no lo son**: el rango es de **7.5 puntos porcentuales**. Escribir que eso está "dentro de límites reducidos" sería afirmar algo que la tabla contradice a simple vista.
- ✅ Sí sostuve la conclusión pedida (robustez), pero **por otro camino**: ninguna de las ocho configuraciones baja del 90%, y ninguna colapsa. La robustez se apoya en el **piso**, no en la estrechez del rango.

Esto obligó a alinear otros dos pasajes, que originalmente también daban a entender uniformidad:
- **FINAL L948** (§6.1) — ahora dice "por encima del 90%, con una variación máxima de 7.5 puntos porcentuales".
- **FINAL L1007** (§7) — la conclusión ya no reclama uniformidad; argumenta que la mejor combinación surge de la integración conjunta de los tres factores.

**Tu decisión:** si tu director prefiere la redacción original, hay que cambiar el argumento, no el número. Pero un revisor que mire la Tabla 12 verá los 7.5 puntos, y "límites reducidos" quedaría difícil de defender.
