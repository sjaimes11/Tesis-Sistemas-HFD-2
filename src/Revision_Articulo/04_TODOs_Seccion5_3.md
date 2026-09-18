# TODOs de §5.3 "Capacidad de detección de la arquitectura propuesta" — resueltos

Ubicación en el paper: **Sección 5 (Results) → §5.3**, los dos párrafos que siguen a la Tabla 12 (`tab:detection_capability`).

## Datos base (accuracy federada real por escenario)
E1 93.81 · E2 92.98 · E3 96.33 · E4 90.00 · E5 96.04 · E6 93.89 · E7 97.50 · E8 94.44

Efectos de factor (medias):
- **Modelo:** CNN 95.5% vs MLP 93.3% → CNN +2.2
- **Seguridad:** ASCON 95.9% vs PLAIN 92.8% → ASCON +3.1 (par a par: +0.8 / +6.3 / +2.2 / +3.1)
- **Topología (Fog):** +0.4 promedio, no uniforme → MLP·ASCON +2.5, CNN·ASCON +1.5, CNN·PLAIN +0.6, MLP·PLAIN −3.0
- **Rango global:** 90.0 (E4) → 97.5 (E7) = 7.5 puntos; todas ≥ 90%.

## Primer TODO (síntesis de factores)

```latex
Los resultados resumidos en la Tabla~\ref{tab:detection_capability} muestran que ninguno de los tres factores compromete la capacidad de detección del sistema. En cuanto al modelo de inferencia, las configuraciones basadas en CNN alcanzan una exactitud promedio del 95.5\%, ligeramente superior a la obtenida por las variantes MLP (93.3\%), lo que confirma que ambas arquitecturas TinyML preservan un desempeño equivalente pese a su diferente complejidad. Respecto a la topología de aprendizaje, la incorporación de la capa \textit{Fog} produce un efecto favorable pero no uniforme: mejora la exactitud en las configuraciones protegidas con ASCON (+2.5 puntos en MLP y +1.5 en CNN) y en CNN sin protección (+0.6 puntos), mientras que en la combinación MLP sin protección introduce una ligera reducción (\(-3.0\) puntos), lo que indica que el beneficio de la agregación intermedia depende de la estabilidad de los mensajes intercambiados. Finalmente, la comparación entre configuraciones protegidas y no protegidas muestra que la incorporación de ASCON no degrada el desempeño: en las cuatro parejas equivalentes, las variantes con ASCON igualan o superan a sus contrapartes PLAIN, con diferencias de entre +0.8 y +6.3 puntos de exactitud. No obstante, este resultado debe interpretarse con cautela, ya que la evaluación no incluyó escenarios con fallos o mensajes corruptos inducidos; por lo tanto, la evidencia permite afirmar que la protección criptográfica es compatible con el proceso de aprendizaje, pero no que lo mejore de manera intrínseca. Los valores de \textit{precision}, \textit{recall} y \textit{F1-score}, determinados por el modelo de inferencia desplegado, permanecen prácticamente constantes (\(\approx\)90.8\% de \textit{F1-score} ponderado en ambas arquitecturas), lo que refuerza la homogeneidad del comportamiento de clasificación entre configuraciones.
```

## Segundo TODO (rango de variación)

```latex
Más allá del efecto individual de cada decisión de diseño, el aspecto más relevante corresponde al comportamiento global de la arquitectura. Las ocho configuraciones evaluadas mantienen una exactitud comprendida entre el 90.0\% (E4, MLP--\textit{Edge--Fog--Cloud}--PLAIN) y el 97.5\% (E7, CNN--\textit{Edge--Fog--Cloud}--ASCON), lo que sitúa la totalidad de las variantes por encima del umbral del 90\% y acota la variación máxima a 7.5 puntos porcentuales. Esta dispersión, limitada y sin ninguna configuración que colapse el desempeño, demuestra que la capacidad de detección se preserva de forma consistente en todo el espacio de diseño explorado. La mejor combinación (CNN con agregación \textit{Fog} y protección ASCON) y la más modesta difieren únicamente en la elección conjunta del modelo, la topología y el mecanismo de protección, sin que ninguna de estas decisiones, por sí sola, reduzca la exactitud por debajo del rango operativo. En consecuencia, la capacidad de detección constituye una propiedad robusta de la arquitectura propuesta y no depende exclusivamente de una combinación particular del modelo de inferencia, de la estrategia de agregación o del mecanismo de protección utilizado.
```

> Redacción honesta: el rango real es 7.5 puntos (no "muy reducido"). La robustez se sostiene en que ninguna variante baja de 90%, no en que sean idénticas.
