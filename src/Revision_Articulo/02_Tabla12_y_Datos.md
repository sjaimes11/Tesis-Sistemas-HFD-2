# Tabla 12 (RQ1) — datos reales + P/R/F1 proxy

## Origen de los números

| Columna | Fuente | Naturaleza |
|---|---|---|
| **Accuracy** por escenario | `executive_summary.json` → `avg_last_global_accuracy` por experimento (= Tabla 4‑6 de la tesis) | **Real, medido** en las corridas federadas (58 intentos) |
| **Precision / Recall / F1** | Recalculado con los modelos offline `MLP_output/ids_3class.keras` y `cnn1d_outputs/ids_3class.keras` sobre el 20% de test estratificado (51 255 flujos) | **Proxy real** del modelo desplegado, **no** por escenario federado |

Validación del proxy: MLP offline dio **Acc 90.72 / F1w 90.82** y CNN **90.70 / 90.80**, que reproducen la Tabla 4‑3 de la tesis (0.9060 / 0.9051). El pipeline es correcto.

### Valores proxy (weighted, estilo tesis)

| Modelo | Precision | Recall | F1 |
|---|---|---|---|
| MLP | 92.11 | 90.72 | 90.82 |
| CNN‑1D | 92.09 | 90.70 | 90.80 |

(Macro, por si se prefiere: MLP P=88.52 R=90.01 F1=88.00 · CNN P=88.50 R=89.98 F1=87.98.)

Como P/R/F1 son del modelo (no del escenario), **las cuatro variantes MLP (E1–E4) comparten P/R/F1**, y lo mismo las cuatro CNN (E5–E8). Es honesto y así se documenta.

## ⚠️ Caveat que debes revisar con el profesor

En una misma fila, la **Accuracy federada** (hasta 97.5%) y el **F1 offline** (≈90.8%) provienen de mediciones distintas (buffer de gateway vs test set global). Verás filas como E7 con Accuracy 97.50 y F1 90.80, que parecen inconsistentes. La nota al pie lo explica, pero hay dos alternativas más limpias si el profesor lo prefiere:

- **(A, implementada)** Accuracy real + P/R/F1 proxy con nota al pie.
- **(B)** Usar también la *accuracy offline* del modelo (90.72 MLP / 90.70 CNN) en toda la fila para consistencia interna, y reportar la accuracy federada por escenario aparte (ya está en la Tabla 13 / §5.4 conceptualmente).
- **(C)** Re‑correr los 8 escenarios registrando la matriz de confusión global por ronda (única vía para P/R/F1 *por escenario* reales).

---

## LaTeX listo para pegar (reemplaza el `\begin{table*}...\end{table*}` de la Tabla 12)

```latex
\begin{table*}[ht]
\centering
\caption{Detection performance obtained for the eight experimental configurations evaluated in the proposed architecture.}
\label{tab:detection_capability}
\footnotesize
\begin{tabular}{lccccccc}
\toprule
&
\multicolumn{3}{c}{\textbf{Experimental Configuration}}
&
\multicolumn{4}{c}{\textbf{Detection Performance}}
\\
\cmidrule(lr){2-4}
\cmidrule(lr){5-8}
\textbf{Scenario} & \textbf{Inference Model} &  \textbf{Architecture} & \textbf{Protection} & \textbf{Accuracy\textsuperscript{a}} & \textbf{Precision\textsuperscript{b}} & \textbf{Recall\textsuperscript{b}} & \textbf{F1-score\textsuperscript{b}}
\\
\midrule
E1 & MLP & Edge--Cloud      & ASCON & 93.81 & 92.11 & 90.72 & 90.82 \\
E2 & MLP & Edge--Cloud      & PLAIN & 92.98 & 92.11 & 90.72 & 90.82 \\
E3 & MLP & Edge--Fog--Cloud & ASCON & 96.33 & 92.11 & 90.72 & 90.82 \\
E4 & MLP & Edge--Fog--Cloud & PLAIN & 90.00 & 92.11 & 90.72 & 90.82 \\
E5 & CNN & Edge--Cloud      & ASCON & 96.04 & 92.09 & 90.70 & 90.80 \\
E6 & CNN & Edge--Cloud      & PLAIN & 93.89 & 92.09 & 90.70 & 90.80 \\
E7 & CNN & Edge--Fog--Cloud & ASCON & 97.50 & 92.09 & 90.70 & 90.80 \\
E8 & CNN & Edge--Fog--Cloud & PLAIN & 94.44 & 92.09 & 90.70 & 90.80 \\
\bottomrule
\end{tabular}
\vspace{2mm}
\footnotesize{\textit{Note:} All values in \%. \textsuperscript{a}~Accuracy is the mean final global accuracy measured on-device across the federated runs of each scenario (58 runs, RQ1 runtime instrumentation). \textsuperscript{b}~Precision, Recall and F1-score (weighted) are computed offline on the held-out 20\% stratified test set (51{,}255 flows) using the deployed MLP and CNN-1D models, since the federated runtime logged only global accuracy and loss; therefore these three metrics depend on the inference model and are shared by its four scenarios.}
\end{table*}
```

Si eliges la opción **(B)** (consistencia interna), cambia la columna Accuracy a `90.72` para E1–E4 y `90.70` para E5–E8, y mueve la accuracy federada por escenario a una tabla/discusión aparte.
