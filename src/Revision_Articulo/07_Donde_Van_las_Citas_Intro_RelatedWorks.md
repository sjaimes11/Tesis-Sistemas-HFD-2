# Dónde va cada cita — Introduction y Related Works (TODO-REF 1–16)

**Estado: NO aplicado al `.tex`.** El `.tex` vive en Overleaf y nunca lo edité; esto es la lista de ubicaciones para aplicarlo a mano. Los números de línea de Overleaf no los conozco, así que cada fila da el párrafo y las últimas palabras de la frase donde va la cita. Busca esas palabras con Ctrl+F.

**Esta tabla es la versión definitiva.** Corrige tres inconsistencias entre `06_TODO_REF_Resueltos.md` y `paper_revisado_CAMBIOS.md` (TODO 7, 13 y 14). Las entradas de `\bibitem` no cambian.

Referencias verificadas contra Crossref el 2026-09-18: Dritsas2025, Cooray2025, Ma2024, Sun2026, Alshammari2026, Fusco2025 (más las ya verificadas antes: Lim2025, Tamuka2026, HeydariSurvey2025, Sorescu2025, y Bartoli2025 en arXiv).

Encaje: **sólido** = la referencia trata justo lo que pide el TODO · **razonable** = trata el tema de cerca pero no de forma específica · **débil** = no encontré una referencia específica (ver notas).

## Introduction

| TODO | Párrafo | La frase termina en… | Cita final | Encaje |
|---|---|---|---|---|
| 1 | P1 "El crecimiento acelerado…" | "…ataques cada vez más sofisticados" | `\cite{Ni2024,Amanullah2020,Bagaa2020,Lim2025}` | razonable |
| 2 | P2 "Los enfoques tradicionales…" | "…en cada nivel de la infraestructura" | `\cite{SaravanaBalaji2023,Ni2024,Dini2024,Tamuka2026}` | sólido |
| 3 | P4 "No obstante, la integración…" | "…dentro de una misma infraestructura" (hoy sin cita) | añadir ` \cite{Alshammari2026,Dritsas2025}` antes del punto | sólido / razonable |

## §2.1 TinyML-Based Intrusion Detection

| TODO | Párrafo | La frase termina en… | Cita final | Encaje |
|---|---|---|---|---|
| 4 | P1 "La evolución de los sistemas de detección…" | "…altamente dinámicos" | `\cite{Ni2024,Amanullah2020,Bagaa2020,SaravanaBalaji2023,Tamuka2026}` | razonable |
| 5 | P3 "La consolidación de TinyML también modificó…" | "…del entorno de ejecución" | `\cite{Tekin2023,Amuthadevi2025,Dini2024,Heydari2024,HeydariSurvey2025,Bartoli2025,Banbury2020}` | sólido |
| 6 | P4 "No obstante, la capacidad de realizar inferencia local…" | "…disponibles en microcontroladores" | `\cite{Ficco2024,Ramadan2025,HeydariSurvey2025,Warden2019}` | **débil** |

## §2.2 Federated Learning for Distributed IDS

| TODO | Párrafo | La frase termina en… | Cita final | Encaje |
|---|---|---|---|---|
| 7 | P1 "El desplazamiento de la inferencia…" | "…sobre infraestructuras Edge" | `\cite{McMahan2017,Imteaj2022,Ramadan2025,Dritsas2025}` | **débil** |
| 8 | P3 "La incorporación del aprendizaje federado modificó…" | "…un elevado grado de variabilidad" | `\cite{Imteaj2022,Hajj2023,Chaurasia2024,Albanbay2025,Ramadan2025,Cooray2025,Ma2024}` | razonable |
| 9 | P4 "A medida que el número de participantes aumenta…" | "…infraestructuras IoT de gran escala" | `\cite{Ficco2024,Tran2026,Ramadan2025,Ma2024,Cooray2025}` | razonable |
| 10 | P5 "Aunque el aprendizaje federado evita compartir…" | "…contra el modelo global" | `\cite{Imteaj2022,Hajj2023,Dritsas2025,Cooray2025}` | razonable / débil |

## §2.3 Lightweight Cryptography

| TODO | Párrafo | La frase termina en… | Cita final | Encaje |
|---|---|---|---|---|
| 11 | P1 "La evolución de los IDS hacia arquitecturas distribuidas…" | "…tráfico inspeccionado por el propio sistema" (hoy sin cita) | añadir ` \cite{Alshammari2026,Dritsas2025}` antes del punto | razonable / débil |
| 12 | P2 "Las restricciones computacionales…" | "…manteniendo un costo computacional reducido" | `\cite{NISTIR8454,NIST800232,Gusita2025,Kaur2025,Sorescu2025}` | sólido |
| 13 | P4 "La literatura reciente ha estudiado ampliamente…" | — | **No añadir cita a P4.** Insertar el párrafo nuevo de GreenShield justo después de P4 (Grupo 5.1 de `paper_revisado_CAMBIOS.md`); ese párrafo ya cita `Alshammari2026`. | razonable |
| 14 | P5 "En consecuencia, aunque la criptografía ligera…" | dentro de la frase: "…Aspectos como el costo temporal del cifrado" | escribir "…el costo temporal del cifrado `\cite{Sorescu2025}`, el incremento del tráfico…" | razonable / débil |

Por qué el 13 no lleva cita al final de P4: esa frase dice que la mayoría de los trabajos *no* analiza el cifrado dentro del aprendizaje distribuido. Citar ahí a Alshammari, que sí lo hace, la contradiría.

## §2.4 Research Gap

| TODO | Párrafo | La frase termina en… | Cita final | Encaje |
|---|---|---|---|---|
| 15 | P2 "A pesar de estos avances…" | reemplazar desde "Aunque existen propuestas que combinan parcialmente…" hasta "…continúa siendo limitada." | usar el texto del Grupo 5.2 (cita `Alshammari2026`, `Tran2026`, `Ramadan2025`) | sólido |
| 16 | P3 "Esta limitación no es únicamente tecnológica…" | "…escenarios de IoT reales" | añadir ` \cite{Alshammari2026,Cooray2025}` antes del punto | razonable |

## Notas sobre los encajes débiles

- **TODO 6 y 7** (aprendizaje continuo / adaptación dinámica): no encontré una referencia específica sobre *continual learning* ni *on-device adaptation* en IDS IoT. Lo que puse cubre TinyML y FL en IoT en general. Si no quieres una cita genérica ahí, la búsqueda queda abierta y la hago si me dices.
- **TODO 10** (ataques de inferencia, poisoning, privacidad de gradientes): Dritsas y Cooray son surveys generales de FL. La búsqueda mostró surveys específicos de seguridad y privacidad en FL, pero no verifiqué sus metadatos, así que no los incluí.
- **TODO 11 y 14**: son citas de contexto, no de evidencia directa.

## Pendiente fuera de esta tabla

Los TODO 17–29 están en la Sección 3 (Proposed Architecture) y tampoco están aplicados. Su mapeo está en `paper_revisado_CAMBIOS.md`, Grupo 4.

Para que lo aplique directamente: sube el `.tex` actual al repo o pégamelo completo, y edito el archivo en vez de darte parches.
