# Section 3: Results

## 3.1 Overall Predictive Performance
All models (M1 through M5) were evaluated on the held-out test cohort ($N = 20,000$ patients) at the validation-locked decision threshold ($th = 0.60$). Table 1 presents the comparative performance across discrimination, precision, calibration, early-warning lead time, and PhysioNet utility score.

### Table 1: Overall Performance Comparison Across Models (Held-Out Test Cohort, N=20,000)
| Model | Architecture | AUROC | AUPRC | F1 | Precision | Recall | ECE | Lead Time | $\ge$6h | $\ge$1h | FPR/h | Utility |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **M1** | XGBoost Baseline | 0.8420 | 0.2650 | 0.2810 | 0.1840 | 0.5820 | 0.0850 | 3.1 h | 22.4% | 41.2% | 0.0480 | -1.4200 |
| **M2** | Plain Transformer | 0.9265 | 0.3540 | 0.3420 | 0.2250 | 0.6150 | 0.0520 | 4.2 h | 29.8% | 48.5% | 0.0310 | -1.1510 |
| **M3** | **Time-Aware Transformer** | **0.9617** | **0.4231** | **0.4110** | **0.3099** | 0.6103 | **0.0407** | **5.7 h** | **37.6%** | **56.5%** | **0.0183** | **-0.9535** |
| **M4** | Organ Hybrid / MoE | 0.9412 | 0.3180 | 0.2640 | 0.1620 | 0.6940 | 0.0780 | 8.6 h | 34.2% | 52.8% | 0.0340 | -1.8420 |
| **M5** | Multi-Hybrid Network | 0.9358 | 0.2751 | 0.1997 | 0.1158 | **0.7251** | 0.0959 | 12.0 h | 39.3% | 56.2% | 0.0580 | -2.5556 |

As shown in Table 1, the gradient boosted decision tree baseline (M1) achieved an AUROC of 0.8420 and an AUPRC of 0.2650. Replacing static window features with a 3-layer Causal Transformer operating on imputed values (M2) increased AUROC to 0.9265 ($\Delta \text{AUROC} = +0.0845$) and AUPRC to 0.3540 ($\Delta \text{AUPRC} = +0.0890$).

Incorporating continuous frequency temporal embeddings (Time2Vec) and missingness masks into the Transformer backbone (M3) yielded the highest discrimination across all models, achieving an **AUROC of 0.9617** (95% CI: `[0.9495, 0.9727]`) and an **AUPRC of 0.4231** (95% CI: `[0.3359, 0.5185]`). Relative to the plain Transformer baseline (M2), M3 improved AUROC by +0.0352 and AUPRC by +0.0691. Figure 1 and Figure 2 illustrate the comparative AUROC and AUPRC values across all five models.

## 3.2 Contribution of Temporal and Observation Information
To isolate the individual effects of continuous elapsed time deltas ($\boldsymbol{\Delta t}$) and binary observation masks ($\mathbf{m}$), we conducted a four-variant ablation study within the frozen M3 architecture at operating threshold $th = 0.60$. Table 2 summarizes the ablation results.

### Table 2: M3 Component Ablation Comparison
| Variant | Values ($\mathbf{v}$) | Mask ($\mathbf{m}$) | Time Delta ($\boldsymbol{\Delta t}$) | AUROC | AUPRC | F1 | Precision | Recall | ECE | Lead Time | FPR/h | Utility |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **M2 / Values-Only** | YES | NO | NO | 0.9265 | 0.3540 | 0.3420 | 0.2250 | 0.6150 | 0.0520 | 4.2 h | 0.0310 | -1.1510 |
| **M3-Time+Delta** | YES | NO | YES | 0.9480 | 0.3890 | 0.3780 | 0.2650 | 0.6020 | 0.0460 | 5.2 h | 0.0240 | -1.0200 |
| **M3-Time+Mask** | YES | YES | NO | 0.9420 | 0.3720 | 0.3610 | 0.2480 | 0.6150 | 0.0490 | 4.8 h | 0.0280 | -1.0800 |
| **M3-Full (Primary)** | YES | YES | YES | **0.9617** | **0.4231** | **0.4110** | **0.3099** | 0.6103 | **0.0407** | **5.7 h** | **0.0183** | **-0.9535** |

The ablation results demonstrate distinct incremental contributions from both components:
1. **Isolated Effect of Continuous Time Deltas ($\boldsymbol{\Delta t}$):** Adding Time2Vec continuous time deltas to the values-only baseline (M2 $\to$ M3-Time+Delta) increased AUROC from 0.9265 to 0.9480 ($\Delta = +0.0215$), increased AUPRC from 0.3540 to 0.3890 ($\Delta = +0.0350$), extended mean early warning lead time from 4.2 hours to 5.2 hours (+1.0 hour), and reduced FPR/h from 0.0310 to 0.0240 (-0.0070).
2. **Isolated Effect of Observation Masks ($\mathbf{m}$):** Adding binary missingness masks to the values-only baseline (M2 $\to$ M3-Time+Mask) increased AUROC from 0.9265 to 0.9420 ($\Delta = +0.0155$), increased precision from 0.2250 to 0.2480 ($\Delta = +0.0230$), extended mean lead time from 4.2 hours to 4.8 hours (+0.6 hours), and reduced FPR/h from 0.0310 to 0.0280 (-0.0030).
3. **Incremental Mask Effect over Time Deltas:** Adding observation masks to the Time+Delta model (M3-Time+Delta $\to$ M3-Full) further increased AUROC from 0.9480 to 0.9617 ($\Delta = +0.0137$), increased AUPRC from 0.3890 to 0.4231 ($\Delta = +0.0341$), increased precision from 0.2650 to 0.3099 ($\Delta = +0.0449$), extended mean lead time from 5.2 hours to 5.7 hours (+0.5 hours), and further lowered FPR/h from 0.0240 to 0.0183 (-0.0057).

Figure 3 displays the ablation AUROC progression, while Figure 9 details the incremental component contributions over the baseline.

## 3.3 Early-Warning Performance and Clinical Trade-offs
Early warning capability was evaluated using mean lead time prior to clinical sepsis onset, early detection rates at $\ge 6$ hours and $\ge 1$ hour prior to onset, and false positive rates per patient-hour (FPR/h).

M3 achieved a mean early warning lead time of **5.7 hours** (95% CI: `[5.0, 6.5]` hours) with a 37.6% $\ge$6-hour early warning rate and a 56.5% $\ge$1-hour early warning rate. Crucially, M3 maintained the lowest false positive rate among all neural architectures at **0.0183 FPR/hour** (1.83% false alarms per patient-hour).

Figure 6 illustrates the trade-off between mean lead time and sensitivity (recall). While M4 (8.6h lead time, 69.4% recall) and M5 (12.0h lead time, 72.5% recall) achieved longer lead times and higher raw sensitivities, Figure 7 demonstrates that this earlier alerting behavior came at the expense of substantially higher false alarm rates (M4: 0.0340 FPR/h; M5: 0.0580 FPR/h).

## 3.4 Calibration Performance
Model calibration was evaluated using Expected Calibration Error (ECE) across 10 reliability bins on the test cohort.

M3 demonstrated the lowest calibration error among all evaluated architectures with an **ECE of 0.0407** (4.07%). By comparison, M2 achieved an ECE of 0.0520, M1 achieved 0.0850, M4 achieved 0.0780, and M5 achieved 0.0959. Figure 8 displays the ECE calibration error comparison across all models.

## 3.5 Architectural Exploration
Table 3 compares the primary compact model (M3) against the exploratory multi-branch hybrid architectures (M4 and M5).

### Table 3: Architectural Exploration Comparison (M3 vs. M4 vs. M5)
| Model | Architecture | Parameters | AUROC | AUPRC | F1 | Precision | Recall | ECE | Lead Time | FPR/h | Utility |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **M3** | Time-Aware Transformer | **163,841** | **0.9617** | **0.4231** | **0.4110** | **0.3099** | 0.6103 | **0.0407** | 5.7 h | **0.0183** | **-0.9535** |
| **M4** | Organ Hybrid / MoE | 198,433 | 0.9412 | 0.3180 | 0.2640 | 0.1620 | 0.6940 | 0.0780 | 8.6 h | 0.0340 | -1.8420 |
| **M5** | Multi-Hybrid Network | 224,713 | 0.9358 | 0.2751 | 0.1997 | 0.1158 | **0.7251** | 0.0959 | 12.0 h | 0.0580 | -2.5556 |

As detailed in Table 3:
- **M4 (Organ Hybrid / MoE):** Adding 6 organ subsystem PATE encoders and prefix token injection increased parameter count to 198,433. Compared to M3, M4 achieved higher recall (0.6940 vs. 0.6103) and longer lead time (8.6h vs. 5.7h), but lower AUROC (0.9412 vs. 0.9617), lower AUPRC (0.3180 vs. 0.4231), lower precision (0.1620 vs. 0.3099), and lower utility score (-1.8420 vs. -0.9535).
- **M5 (Multi-Hybrid Network):** Splitting inputs into disjoint branch encoders and routing through 3 temporal experts via an MoE router increased parameter count to 224,713. M5 achieved the highest recall (0.7251) and longest lead time (12.0h), but lowest precision (0.1158), highest FPR/h (0.0580), and lowest utility score (-2.5556).

Figure 10 illustrates the PhysioNet Utility score comparison, confirming that M3 achieved the optimal balance of early warning utility.

## 3.6 Discrimination and Operating Characteristics
Precision-Recall (PR) and Receiver Operating Characteristic (ROC) curves across all models are presented in Figure 4 and Figure 5, respectively:
- **PR Curves (Figure 4):** M3 dominates the precision-recall envelope across all recall operating points, maintaining a precision of $>0.30$ up to $0.60$ recall, whereas M4 and M5 precision drops below $0.20$ beyond $0.50$ recall.
- **ROC Curves (Figure 5):** M3 maintains superior true positive rates across the low false positive rate regime ($\text{FPR} < 0.10$), yielding an overall AUROC of 0.9617 compared to M4 (0.9412) and M5 (0.9358).
