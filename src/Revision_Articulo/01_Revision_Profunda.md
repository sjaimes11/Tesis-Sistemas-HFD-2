# Revisión profunda — "An Edge–Fog–Cloud Architectural Framework for Distributed IoT Intrusion Detection Based on Hierarchical Federated Learning"

Revisión del manuscrito MDPI (versión julio 22, 2026) contra la tesis `Rev1__Tesis_Sistemas_7`. Organizada por severidad y por sección. Marco cada punto como:
🔴 **Bloqueante** (hay que resolverlo antes de enviar) · 🟠 **Importante** · 🟡 **Menor / estilo**.

---

## 0. Resumen ejecutivo

El artículo re-enmarca la tesis de "sistema construido y medido" a "framework arquitectónico con dos principios de diseño" (*equilibrio arquitectónico* e *integración arquitectónica*). El giro es coherente y está bien sostenido en Related Works, Proposed Architecture y Discusión. Los datos cuantitativos que **sí** se reportan (latencia cripto, overhead, tiempo/ronda, accuracy por escenario) coinciden con la tesis. Los problemas críticos son tres:

1. **Tabla 12 vacía** (resultado central de RQ1) — resuelto en el doc `02` con datos reales + proxy documentado.
2. **RQ4 (overhead cripto) está ubicado dentro de la sección de Metodología (§4.8), no en Resultados (§5)** — descuadre estructural.
3. **Bibliografía con 5 claves duplicadas y cross-references rotas** — resuelto en el doc `03`.

---

## 1. Estructura global y coherencia

🔴 **§4.8 mal ubicada.** "Impacto de la protección criptográfica sobre el aprendizaje federado" (con Tablas 9, 10 y Figuras 6, 7 — que responden RQ4) está dentro de la **Sección 4 (Metodología de evaluación experimental)**, *antes* de la Sección 5 (Results). Consecuencia: la Sección 5 "Results" presenta RQ (modelo, convergencia, detección, eficiencia) pero **no incluye RQ4**, que ya se "gastó" en la metodología. Además §4.8 abre con "La subsección anterior demostró…" refiriéndose a resultados que aún no se han presentado formalmente. **Recomendación:** mover §4.8 completa (texto + Tablas 9/10 + Figuras 6/7) a una subsección de la Sección 5 (p.ej. §5.5 "Cryptographic protection overhead"), después de §5.4.

🟠 **Duplicación conceptual metodología ↔ resultados.** Las subsecciones 4.7.x ("Proceso de aprendizaje federado", "Desempeño en Edge", "Comunicación segura") describen *marcos de evaluación* (Model Quality, Training Dynamics, etc.) y luego la Sección 5 vuelve a narrar lo mismo. Hay solapamiento de prosa. Considerar condensar los "Evaluation Framework" (Figuras 5, 8) para no repetir.

🟠 **Figura 3 vs Figura 4.** Tienen captions distintos ("Experimental evaluation workflow" y "Experimental protocol workflow") pero comparten el mismo `\label` (ver doc `03`, cross-ref). Verificar que sean **dos diagramas distintos**; si son el mismo, dejar uno.

🟡 Numeración: la Sección 5 se titula "Results" pero la Sección 4 ya contiene resultados (§4.8). Al reubicar §4.8 esto se corrige.

---

## 2. Abstract y título

🟡 Abstract sólido y autocontenido. Coherente con las 5 dimensiones (RQ1–RQ5). Sin cambios de fondo.

🟡 **RNN en la lista de abreviaturas** pero nunca se usa RNN en el texto (la arquitectura es CNN‑1D con Conv1D, no recurrente). Eliminar `RNN` de `\abbreviations`.

🟡 El título y abstract están en inglés y el cuerpo en español — esperado en esta fase (el profesor traducirá después). Al traducir, unificar `MLP`/`RN`: la tesis usa "RN/MLP" indistintamente; el artículo debe usar **solo MLP** (ya lo hace en Tabla 2 y 12, pero las Tablas 9/10/13 y los nombres de figura usan "RN"/"NoFog"). Ver §5 de este doc.

---

## 3. Introduction

🟢 Buen embudo (IoT → superficie de ataque → IDS → Edge‑Fog‑Cloud → 3 tecnologías → brecha → propuesta). Coherente con el nuevo enfoque de "integración".

🟠 Hay 4 comentarios `% TODO-REF` pidiendo referencias recientes (2025–2026) de tendencias IoT/IIoT, arquitecturas Edge‑Fog‑Cloud para ciberseguridad, e integración IA+FL+seguridad. Son huecos reales de literatura; conviene llenarlos antes de enviar (una revista suele exigir actualidad bibliográfica). Puedo proponer candidatos si quieres.

🟡 `\cite{SaravanaBalaji2023,Ni2024,Dini2024}` para sustentar "Edge‑Fog‑Cloud": Saravana Balaji es edge sanitario con DBN y Ni2024 es survey de IIoT‑ML; encajan de forma indirecta. Aceptable, pero una cita específica de arquitecturas E‑F‑C reforzaría.

---

## 4. Related Works

🟢 La organización (TinyML → FL → criptografía ligera → brecha) es clara y termina bien en la *research gap*. Es la mejor parte del re‑enmarque.

🟠 **Consistencia de la tabla de comparación (Tabla 14).** La fila "Tran et al." marca IDS = ✗. En la tesis, EdgeTrust‑Shard (Tran) se cita como trabajo de FL+blockchain robusto a bizantinos; que su columna IDS quede en ✗ es defendible (no es un IDS per se), pero verifica que esa lectura sea intencional, porque contrasta con cómo se cita en el resto del texto.

🟠 Muchos `% TODO-REF` sin resolver (≈12 en Related Works). No son errores, pero son deuda de literatura que un revisor notará. Prioriza los de la *research gap* (§2.4).

🟡 "deteccion de intrusiones" sin tilde aparece varias veces desde aquí en adelante (§2, §6, §7). Corregir a "detección".

---

## 5. Proposed Architecture (Sección 3)

🟢 Descripción por capas (Edge/Fog/Cloud) clara y consistente con la tesis (ESP32‑S3, RPi 4, PC coordinador, FedAvg, ASCON‑128 transversal).

🟠 **Terminología de topología.** El texto y las Tablas 2/12 usan **"Edge–Cloud" vs "Edge–Fog–Cloud"** para NoFOG vs FOG. Pero las Tablas 9, 10, 13 y los nombres de figura usan **"NoFog"/"Fog"**, y la tesis usaba "NoFOG/FOG". Unifica: recomiendo **"Edge–Cloud"/"Edge–Fog–Cloud"** en prosa/tablas de detección y aclarar una sola vez que equivalen a NoFog/Fog. Que dos vocabularios convivan confunde al lector.

🟠 **"scan"/"scan_A"/"Network Scan".** La tesis usa `scan_A`; §3.2 dice clasificar como "normal, MQTT brute‑force o scan"; la Tabla 3 dice "Network Scan". Unifica el nombre de la tercera clase en todo el documento (sugiero "Network Scan (scan_A)" en la primera mención y "scan" después).

🟡 §3.2 (Edge) menciona 13 características pero no remite a la Tabla 4 hasta la metodología; un `\ref{tab:flow_features}` temprano ayudaría.

🟡 Falta la **arquitectura interna del modelo MLP** (la Figura 3‑2 de la tesis: 13→32→16→8→3, capas federadas W3/b3/W4/b4). El artículo describe FL pero nunca muestra qué se federa (163 parámetros / 2 últimas capas). Recomiendo recuperar esa figura o al menos una frase: "solo las dos capas densas finales (163 parámetros, ≈652 B) se intercambian por ronda". Es un dato distintivo y barato de añadir.

---

## 6. Metodología (Sección 4)

🟢 Diseño factorial 2³ (Tabla 6) bien planteado; mapeo E1–E8 consistente con la tesis (RN=MLP). El protocolo experimental y la plataforma están bien descritos.

🔴 **Cross-reference rota:** §4.7 (Métricas) dice "definidas en la Sección~\ref{sec:evaluation_methodology}" pero ese `\label` no existe → sale "Sección **??**". Fix en doc `03`.

🔴 **Doble `\label{fig:experimental_workflow}`** en Figura 3 y Figura 4. Fix en doc `03`.

🟠 **RQ3 pide CPU/Memory Utilization que no se midió** (Tabla 8, §4.7.3). Según tu decisión, lo dejamos como está por ahora; pero ten presente que RQ3 promete indicadores (utilización de CPU/memoria) que la Tabla de resultados de Edge no podrá llenar con datos reales. Si un revisor lo pide, habrá que instrumentar el ESP32 o reformular RQ3 a huella de memoria + latencia de inferencia (que sí existen).

🟠 **TODOs de metodología sin cerrar:** entorno software (TF, TFLM, versión de Python), specs definitivas del servidor, valor de buffer `Ns=30` y regla de etiquetado heurístico, criterio de terminación (rondas fijas vs convergencia). Todos existen en la tesis (§3.2–3.6) — es cuestión de trasladarlos. Puedo hacerlo si quieres.

🟡 §4.7.4 lista CPU/Memory Utilization también para "Resource Overhead" del mecanismo cripto — mismo comentario que RQ3.

---

## 7. Results (Sección 5)

🔴 **Tabla 12 vacía (XX.XX).** Es el resultado principal de RQ1. Datos reales + P/R/F1 proxy en doc `02`. **Caveat importante a documentar:** la columna Accuracy proviene de las corridas federadas (medida sobre buffers de gateway) y llega hasta 97.5%, mientras que P/R/F1 proxy provienen de los modelos offline sobre el test set (≈90.8% F1w). En una misma fila, Accuracy=97.5 con F1=90.8 se ve inconsistente. La nota al pie lo explicará, pero la alternativa más limpia sería re‑correr los 8 escenarios registrando la matriz de confusión global. Decisión tuya (ver doc `02`).

🟢 §5.1 (selección de modelo, Tabla 11) coincide exactamente con la tesis (RF 99.94, DT 99.87, SVM‑RBF 99.68; MLP 90.60, CNN 90.51). Bien.

🟢 §5.2 (convergencia + estabilidad, Figs 9, 10) y §5.4 (duración/ronda, Fig 11, Tabla 13) coinciden con los CSV reales. Figuras regeneradas en inglés (doc `03` / carpeta `figures_en`).

🟠 §5.3 (Tabla 12) afirma "las métricas de clasificación permanecen consistentemente elevadas" y "variaciones dentro de un rango reducido". Con los datos reales, la Accuracy va de **90.0% (E4) a 97.5% (E7)** — un rango de 7.5 puntos, que no es tan "reducido". Ajustar la redacción para que sea fiel al dato (p.ej. "se mantienen por encima del 90% en todas las variantes, con la mejor combinación FOG+ASCON").

🟠 §5.4 dice que la capa Fog "no produce crecimiento progresivo del costo temporal". Correcto, pero MLP+ASCON+Fog/NoFog (71–73 s) es notablemente más lento que CNN (52–59 s). El texto debería reconocer que **MLP es más lento por ronda que CNN** (contraintuitivo, vale la pena una frase explicándolo: buffers, épocas locales, etc.).

---

## 8. Discusión y Conclusiones (Secciones 6–7)

🟢 Los dos principios ("equilibrio" e "integración" arquitectónicos) están bien argumentados y son el hilo narrativo nuevo. Coherentes con Related Works y Results.

🟠 La Discusión afirma que ASCON no penaliza la convergencia y que la seguridad se integra "sin comprometer" el desempeño. Es defendible con los datos, **pero** — como ya notaba la tesis — no se probó con fallos/mensajes corruptos inducidos, así que no se puede afirmar que ASCON *mejore* el aprendizaje. Mantener el matiz de la tesis (§5.1 conclusiones) para no sobre‑afirmar.

🟡 "deteccion" sin tilde repetido en §6 y §7. Corregir.

🟡 §6.2 (Tabla 14) — misma observación que Related Works sobre la fila Tran/IDS.

---

## 9. Checklist accionable (orden sugerido)

| # | Acción | Severidad | Dónde |
|---|--------|-----------|-------|
| 1 | Llenar Tabla 12 (accuracy real + P/R/F1 proxy + nota) | 🔴 | doc `02` |
| 2 | Arreglar `\label` duplicado Fig 3/4 y `sec:evaluation_methodology` | 🔴 | doc `03` |
| 3 | Quitar 5 `\bibitem` duplicados de la bibliografía | 🔴 | doc `03` |
| 4 | Mover §4.8 (RQ4) a la Sección 5 | 🔴 | §4.8→§5.5 |
| 5 | Sustituir figuras por versiones en inglés | 🟠 | `figures_en/` |
| 6 | Unificar MLP/RN y Edge–Cloud/NoFog y scan/scan_A | 🟠 | global |
| 7 | Ajustar redacción §5.3 y §5.4 al rango real de accuracy | 🟠 | §5.3, §5.4 |
| 8 | Cerrar TODOs de metodología (software, buffer, terminación) | 🟠 | §4.3–4.4 |
| 9 | Decidir RQ3 (CPU/mem) | 🟠 | §4.7.3 |
| 10 | Corregir "deteccion"→"detección", quitar RNN de abreviaturas | 🟡 | global |
| 11 | Resolver `% TODO-REF` de literatura reciente | 🟡 | §1, §2 |
