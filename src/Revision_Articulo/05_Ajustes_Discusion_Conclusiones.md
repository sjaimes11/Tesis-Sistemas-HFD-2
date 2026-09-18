# Ajustes §6.1 (Discusión) y §7 (Conclusiones) — alineados con §5.3

Objetivo: que la Discusión y las Conclusiones no sobre-afirmen "variación mínima / configuraciones equivalentes". El dato real es: todas las variantes ≥90% de exactitud, rango de 7.5 puntos (90.0 E4 → 97.5 E7), mejor = CNN·Edge–Fog–Cloud·ASCON. La robustez se sostiene en "ninguna colapsa", no en "todas iguales".

## §6.1 — Implicaciones arquitectónicas de la propuesta

**ANTES** (final del 2º párrafo):
> …mientras que la evaluación experimental confirmó que es posible preservar una elevada capacidad de detección mediante modelos específicamente diseñados para plataformas con recursos restringidos.

**DESPUÉS:**
```latex
mientras que la evaluación experimental confirmó que es posible preservar una elevada capacidad de detección mediante modelos específicamente diseñados para plataformas con recursos restringidos: las ocho configuraciones evaluadas se mantuvieron por encima del 90\% de exactitud, con una variación máxima de 7.5 puntos porcentuales y el mejor desempeño en la combinación de CNN, agregación \textit{Fog} y protección ASCON.
```

## §7 — Conclusiones

**ANTES** (2º párrafo):
> Consideradas de manera conjunta, estas decisiones mostraron un comportamiento consistente en las diferentes configuraciones experimentales, lo que proporciona evidencia de que la interacción entre los componentes de la arquitectura resulta determinante para su desempeño global.

**DESPUÉS:**
```latex
Consideradas de manera conjunta, estas decisiones preservaron la capacidad de detección en todas las configuraciones experimentales, que se mantuvieron por encima del 90\% de exactitud, y evidenciaron que la interacción entre los componentes de la arquitectura resulta determinante para su desempeño global, como refleja el hecho de que la mejor combinación surja de la integración conjunta del modelo de inferencia, la topología de agregación y el mecanismo de protección, y no de la optimización aislada de ninguno de ellos.
```

## Notas de coherencia (no modificadas)
- Abstract: "maintains consistent detection performance" — aceptable (todas ≥90%).
- §6.1 1er párrafo y §7 1er párrafo: ya hablan de "equilibrio/interacción", alineados.
- §6.3 (Limitaciones): "preservación de la capacidad de detección" — consistente.
