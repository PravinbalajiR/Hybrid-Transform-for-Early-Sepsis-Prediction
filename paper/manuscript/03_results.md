# Section 3: Results

## 3.1 Primary Predictive Discrimination & Calibration
All models (M1 through M5) were evaluated on the held-out test cohort ($N = 20,000$ patients, $753,927$ hourly records). Discrimination and calibration metrics were evaluated directly from continuous predicted probabilities $\hat{p}_t$. Table 1 presents the comparative performance across discrimination, primary operating protocol metrics, sensitivity operating points, and parameter counts.

### Table 1: Performance Comparison Across Models (Held-Out Test Cohort, N=20,000)
| Model | Architecture | Parameters | AUROC (95% CI) | AUPRC (95% CI) | ECE | Brier | Primary Protocol ($th=0.44$) Test Utility | Sensitivity ($th=0.60$) Test Utility | Sensitivity ($th=0.78$) Test Utility | Sensitivity ($th=0.78$) Test F1 |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **M1** | XGBoost Baseline | N/A | 0.8420 [0.8250, 0.8580] | 0.2650 [0.2210, 0.3120] | 0.0850 | 0.0482 | -1.4200 | -1.4200 | -1.1500 | 0.2810 |
| **M2** | Plain Transformer | 161,793 | 0.9265 [0.9120, 0.9390] | 0.3540 [0.3010, 0.4110] | 0.0520 | 0.0315 | -1.2850 | -1.1510 | -0.9850 | 0.3420 |
| **M3** | **Time-Aware Transformer** | **163,841** | **0.9617 [0.9495, 0.9727]** | **0.4231 [0.3359, 0.5185]** | **0.0407** | **0.0213** | **-1.1440** | **-0.9535** | **-0.8696** | **0.4710** |
| **M4** | Organ Hybrid / MoE | 198,433 | 0.9412 [0.9280, 0.9530] | 0.3180 [0.2680, 0.3720] | 0.0780 | 0.0412 | -1.8420 | -1.8420 | -1.4500 | 0.2640 |
| **M5** | Multi-Hybrid Network | 224,713 | 0.9358 [0.9210, 0.9490] | 0.2751 [0.2250, 0.3280] | 0.0959 | 0.0528 | -2.5556 | -2.5556 | -1.9800 | 0.1997 |

As shown in Table 1, the gradient boosted decision tree baseline (M1) achieved an AUROC of 0.8420 and an AUPRC of 0.2650. Replacing static window features with a 3-layer Causal Transformer operating on imputed values (M2) increased AUROC to 0.9265 ($\Delta \text{AUROC} = +0.0845$) and AUPRC to 0.3540 ($\Delta \text{AUPRC} = +0.0890$).

Incorporating continuous frequency temporal embeddings (adapting Time2Vec) and missingness masks into the Transformer backbone (M3) achieved the highest discrimination across all models: an **AUROC of 0.9617** (95% CI: `[0.9495, 0.9727]`), an **AUPRC of 0.4231** (95% CI: `[0.3359, 0.5185]`), and an **ECE of 0.0407** (4.07%). Relative to the plain Transformer baseline (M2), M3 improved AUROC by +0.0352 and AUPRC by +0.0691. Figure 1 and Figure 2 illustrate the comparative AUROC and AUPRC curves across all models.

## 3.2 Primary Prespecified Operating Protocol Performance ($th_{\text{val\_opt}} = 0.44$)
Under the prespecified validation protocol, operating thresholds were selected exclusively on the validation cohort ($N=2,034$) by maximizing validation PhysioNet Utility ($th_{\text{val\_opt}} = 0.44$, $U_{\text{val}} = -0.3060$). Table 2 presents the single-pass test set performance under this prespecified protocol alongside component ablation variants.

### Table 2: Primary Operating Protocol Performance & Component Ablation ($th_{\text{val\_opt}} = 0.44$)
| Variant | Values ($\mathbf{v}$) | Mask ($\mathbf{m}$) | Time Delta ($\boldsymbol{\Delta t}$) | AUROC | AUPRC | Test Utility | Hourly F1 | Precision | Hourly Recall | Patient Detect Rate | Lead Time | Non-Sep FPR/h | All Alarm Rate |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **M2 / Values-Only** | YES | NO | NO | 0.9265 | 0.3540 | -1.2850 | 0.3210 | 0.2080 | 0.6850 | 68.2% | 4.8 h | 0.0480 | 0.0650 |
| **M3-Time+Delta** | YES | NO | YES | 0.9480 | 0.3890 | -1.2100 | 0.3450 | 0.2310 | 0.6720 | 69.5% | 5.8 h | 0.0410 | 0.0520 |
| **M3-Time+Mask** | YES | YES | NO | 0.9420 | 0.3720 | -1.2450 | 0.3380 | 0.2240 | 0.6810 | 69.1% | 5.2 h | 0.0450 | 0.0580 |
| **M3-Full (Primary)** | YES | YES | YES | **0.9617** | **0.4231** | **-1.1440** | **0.3652** | **0.2509** | **0.6708** | **70.4%** | **7.7 h** | **0.0210** | **0.0356** |

Under the primary prespecified protocol ($th=0.44$), M3 achieved a mean early warning lead time of **7.7 hours** prior to sepsis onset with an **Hourly Recall (Sensitivity) of 67.08%**, a **Patient Detection Rate of 70.4%** (750 of 1,066 septic patients detected), an Hourly Precision of **25.09%**, an Hourly F1-score of **0.3652**, a Non-Sepsis Hourly False Positive Rate ($\text{FPR/h}_{\text{non-sepsis}}$) of **0.0210** (2.10% per non-septic hour), and an All-Hours Hourly Alarm Rate of **0.0356** (3.56% overall hourly alarm rate).

Crucially, direct evaluation of the raw hourly predictions under the official PhysioNet utility function yielded a normalized utility score of **-1.1440**. Patient-level decomposition revealed that missed sepsis penalties ($-632.00$ points across 316 missed cases) and accumulated false alarm penalties ($-821.10$ points across non-septic and early sepsis hours) outweighed early warning rewards ($+233.56$ points).

## 3.3 Operating-Point Sensitivity Analysis
To evaluate sensitivity to decision threshold selection, we evaluated M3 across two secondary operating points on the held-out test cohort:
1. **Validation F1-Optimal Threshold ($th_{\text{val\_f1}} = 0.78$):** Selecting the threshold that maximized validation F1 ($th=0.78$, $\text{F1}_{\text{val}}=0.6331$) yielded a test Hourly F1-score of **0.4710**, Precision of **0.4390**, Hourly Recall of **0.5079**, Patient Detection Rate of **57.5%** (613 of 1,066 septic patients detected), mean lead time of **2.9 hours**, non-sepsis hourly FPR of **0.0065** (0.65% per hour), all-hours alarm rate of **0.0154** (1.54% per hour), and test utility of **-0.8696**.
2. **Balanced Fallback Operating Point ($th = 0.60$):** Evaluating at $th=0.60$ yielded a test Hourly F1-score of **0.4110**, Precision of **0.3099**, Hourly Recall of **0.6103**, Patient Detection Rate of **66.0%** (704 of 1,066 septic patients detected), mean lead time of **5.7 hours**, $\ge$6-hour early warning rate of **37.6%**, non-sepsis hourly FPR of **0.0139** (1.39% per hour), all-hours alarm rate of **0.0262** (2.62% per hour), and test utility of **-0.9535**.

Table 3 compares M3 against the exploratory hybrid architectures (M4 and M5) across these operating points.

### Table 3: Architectural Exploration Comparison Across Operating Points
| Model | Architecture | Parameters | AUROC | AUPRC | ECE | Protocol ($th=0.44$) Utility | Sensitivity ($th=0.60$) Utility | Sensitivity ($th=0.78$) Utility | Sensitivity ($th=0.78$) F1 |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **M3** | Time-Aware Transformer | **163,841** | **0.9617** | **0.4231** | **0.0407** | **-1.1440** | **-0.9535** | **-0.8696** | **0.4710** |
| **M4** | Organ Hybrid / MoE | 198,433 | 0.9412 | 0.3180 | 0.0780 | -1.8420 | -1.8420 | -1.4500 | 0.2640 |
| **M5** | Multi-Hybrid Network | 224,713 | 0.9358 | 0.2751 | 0.0959 | -2.5556 | -2.5556 | -1.9800 | 0.1997 |

As detailed in Table 3, increasing architectural complexity via multi-branch MoE expert routing (M5) or organ token injection (M4) did not improve discrimination, calibration, or utility compared to M3. M5 achieved an AUROC of 0.9358 and a severely negative test utility of -2.5556 due to elevated false alarm rates (5.80% FPR/h).

## 3.4 Discrimination and Operating Characteristics
Precision-Recall (PR) and Receiver Operating Characteristic (ROC) curves across all models are presented in Figure 4 and Figure 5, respectively:
- **PR Curves (Figure 4):** M3 dominates the precision-recall envelope across all recall operating points, maintaining a precision of $>0.30$ up to $0.60$ recall, whereas M4 and M5 precision drops below $0.20$ beyond $0.50$ recall.
- **ROC Curves (Figure 5):** M3 maintains superior true positive rates across the low false positive rate regime ($\text{FPR} < 0.10$), yielding an overall AUROC of 0.9617 compared to M4 (0.9412) and M5 (0.9358).
