# NIST Relevance Report - HFL v7

This report is generated from the project CSV results. It separates NIST-backed claims from project-specific benchmark thresholds.

## NIST Scope Correction

- NIST selected the Ascon family for lightweight cryptography standardization for constrained devices in 2023.
- NIST finalized the Ascon-based lightweight cryptography standard in 2025. Present this as selected in 2023 and standardized/finalized later, not as fully finalized in 2023.
- NIST SP 800-22 is for randomness tests of RNG/PRNG bitstreams. It should not be cited as the source for t-tests, Wilcoxon tests, or the n>=30 rule for FL accuracy/loss experiments.
- The `1000 operations`, `p95 <= 50 ms`, and `overhead <= 50%` checks in this folder are project benchmark thresholds, not literal NIST pass/fail requirements.

## ASCON Threshold Summary

| experiment | architecture | attempts | total_ascon_ops | min_ops_per_attempt | attempts_passing_1000_ops | mean_p95_latency_ms | max_p95_latency_ms | mean_overhead_pct | mean_expansion_ratio | passes_total_ops_1000 | passes_all_attempts_1000_ops | passes_p95_latency_50ms | passes_avg_overhead_50pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CNN_FOG_ASCON | CNN_FOG | 7 | 8493 | 40 | 6 | 36.7884 | 42.1082 | 84.532 | 1.84532 | True | False | True | False |
| RN_ASCON | RN | 9 | 13154 | 1380 | 9 | 34.3642 | 34.9413 | 92.5707 | 1.92571 | True | True | True | False |

## Independent Run Sufficiency

| experiment | independent_attempts | observed_round_rows | recommended_independent_attempts | passes_30_independent_attempts | round_rows_at_least_30 | note |
| --- | --- | --- | --- | --- | --- | --- |
| CNN_FOG_ASCON | 7 | 120 | 30 | False | True | Round rows are repeated measures, not fully independent experimental executions. |
| RN_ASCON | 9 | 190 | 30 | False | True | Round rows are repeated measures, not fully independent experimental executions. |
| RN_NO_ASCON | 14 | 289 | 30 | False | True | Round rows are repeated measures, not fully independent experimental executions. |

## RN ASCON vs no-ASCON Statistical Tests

| comparison | metric | n_ascon | n_no_ascon | mean_ascon | mean_no_ascon | mean_diff_ascon_minus_no_ascon | percent_change_vs_no_ascon | ci95_mean_diff_low | ci95_mean_diff_high | shapiro_p_ascon | shapiro_p_no_ascon | selected_test | test_statistic | p_value | alpha | statistically_significant | cohens_d | paired_design | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RN_ASCON_vs_RN_NO_ASCON | edge_processing_p95_ms | 9 | 14 | 4.70175 | 0.166507 | 4.53524 | 2723.75 | 4.47647 | 4.59369 | 0.879744 | 0.0866649 | welch_ttest | 142.243 | 3.90862e-15 | 0.05 | True | 76.2672 | False | Mann-Whitney is used for non-normal independent samples; Wilcoxon signed-rank would require paired runs. |
| RN_ASCON_vs_RN_NO_ASCON | server_ingress_processing_p95_ms | 9 | 14 | 20.3497 | 0 | 20.3497 |  | 17.636 | 23.2243 | 0.97951 |  | mann_whitney_u | 126 | 7.7597e-06 | 0.05 | True | 7.41604 | False | Mann-Whitney is used for non-normal independent samples; Wilcoxon signed-rank would require paired runs. |
| RN_ASCON_vs_RN_NO_ASCON | server_egress_processing_p95_ms | 9 | 14 | 19.4347 | 0.250814 | 19.1839 | 7648.63 | 16.0267 | 23.4626 | 0.0086407 | 0.0405869 | mann_whitney_u | 126 | 8.2462e-05 | 0.05 | True | 5.04646 | False | Mann-Whitney is used for non-normal independent samples; Wilcoxon signed-rank would require paired runs. |
| RN_ASCON_vs_RN_NO_ASCON | edge_payload_bytes_avg | 9 | 14 | 232.762 | 115.287 | 117.475 | 101.897 | 116.923 | 118.146 | 0.169013 | 0.00105539 | mann_whitney_u | 126 | 8.2462e-05 | 0.05 | True | 131.954 | False | Mann-Whitney is used for non-normal independent samples; Wilcoxon signed-rank would require paired runs. |
| RN_ASCON_vs_RN_NO_ASCON | avg_round_duration_sec | 9 | 14 | 61.2549 | 51.0114 | 10.2435 | 20.0809 | 7.80888 | 12.375 | 0.0076499 | 3.66545e-07 | mann_whitney_u | 120 | 0.000360947 | 0.05 | True | 3.34417 | False | Mann-Whitney is used for non-normal independent samples; Wilcoxon signed-rank would require paired runs. |
| RN_ASCON_vs_RN_NO_ASCON | last_global_accuracy | 9 | 14 | 0.946296 | 0.929762 | 0.0165344 | 1.77835 | -0.00529085 | 0.0402115 | 0.148934 | 0.0135973 | mann_whitney_u | 76.5 | 0.391042 | 0.05 | False | 0.559971 | False | Mann-Whitney is used for non-normal independent samples; Wilcoxon signed-rank would require paired runs. |
| RN_ASCON_vs_RN_NO_ASCON | last_global_loss | 9 | 14 | 0.141462 | 0.158135 | -0.0166737 | -10.5439 | -0.0728337 | 0.037372 | 0.417757 | 0.481064 | welch_ttest | -0.576265 | 0.572169 | 0.05 | False | -0.248889 | False | Mann-Whitney is used for non-normal independent samples; Wilcoxon signed-rank would require paired runs. |
| RN_ASCON_vs_RN_NO_ASCON | round_completion_rate | 9 | 14 | 0.703704 | 0.688095 | 0.0156085 | 2.26836 | -0.0642857 | 0.111111 | 3.21749e-07 | 4.88936e-07 | mann_whitney_u | 69.5 | 0.518321 | 0.05 | False | 0.158147 | False | Mann-Whitney is used for non-normal independent samples; Wilcoxon signed-rank would require paired runs. |

## Claim Audit

| claim | status | evidence | source |
| --- | --- | --- | --- |
| ASCON was selected by NIST for lightweight cryptography standardization. | supported_by_nist | NIST selected the Ascon family in 2023 and later finalized an Ascon-based lightweight cryptography standard. | NIST LWC / SP 800-232 |
| SP 800-22 validates n>=30 FL experiment runs. | needs_correction | SP 800-22 is a statistical test suite for random and pseudorandom number generators, not a guideline for FL accuracy/loss experiments. | NIST SP 800-22 |
| At least 1000 ASCON operations were observed. | data_supported | See ascon_threshold_summary_by_experiment.csv. | project CSV metrics |
| At least 30 independent full experiment executions were captured. | not_supported | Current datasets have fewer than 30 independent attempts per experiment. Round rows can be analyzed as repeated measures, not independent executions. | project CSV metrics |
| Average ASCON overhead stays below 50%. | not_supported_or_partial | Small JSON payloads plus base64 envelope can exceed 50% overhead. See threshold tables. | project CSV metrics |

## Figures

- `ascon_operations_threshold`: `C:\Users\VivoBook\Downloads\microproyecto2\microproyecto 2 porque wtf\src\Analisis de Modelos\NIST\analysis_outputs\nist_relevance\ascon_operations_threshold.png`
- `ascon_overhead_threshold`: `C:\Users\VivoBook\Downloads\microproyecto2\microproyecto 2 porque wtf\src\Analisis de Modelos\NIST\analysis_outputs\nist_relevance\ascon_overhead_threshold.png`
- `rn_ascon_vs_no_ascon_edge_p95`: `C:\Users\VivoBook\Downloads\microproyecto2\microproyecto 2 porque wtf\src\Analisis de Modelos\NIST\analysis_outputs\nist_relevance\rn_ascon_vs_no_ascon_edge_p95.png`
- `rn_ascon_vs_no_ascon_percent_change`: `C:\Users\VivoBook\Downloads\microproyecto2\microproyecto 2 porque wtf\src\Analisis de Modelos\NIST\analysis_outputs\nist_relevance\rn_ascon_vs_no_ascon_percent_change.png`

## Official NIST References

- NIST LWC selection of Ascon, 2023: https://www.nist.gov/news-events/news/2023/02/lightweight-cryptography-standardization-process-nist-selects-ascon
- NIST article on Ascon for small devices, 2023: https://www.nist.gov/news-events/news/2023/02/nist-selects-lightweight-cryptography-algorithms-protect-small-devices
- NIST final lightweight cryptography standard announcement, 2025: https://www.nist.gov/news-events/news/2025/08/nist-finalizes-lightweight-cryptography-standard-protect-small-devices
- NIST SP 800-22 randomness test suite: https://www.nist.gov/publications/statistical-test-suite-random-and-pseudorandom-number-generators-cryptographic-1