# Section 4: Discussion

## 4.1 Summary of Main Findings
This study investigated whether explicitly representing physiological values, observation missingness patterns, and continuous temporal intervals within a Transformer architecture improves early sepsis prediction from irregular ICU data. Our findings indicate that a compact Time-Aware Transformer (M3) incorporating continuous frequency time-delta embeddings (adapting Time2Vec; Kazemi et al., 2019) and missingness masks achieves higher predictive discrimination ($\text{AUROC} = 0.9617$, $\text{AUPRC} = 0.4231$) and calibration ($\text{ECE} = 0.0407$) compared to standard tree ensembles (M1), plain Transformer baselines (M2), and multi-branch hybrid architectures (M4 and M5).

However, our controlled component ablations and operational audits demonstrate a fundamental clinical insight: **strong predictive discrimination does not automatically translate into positive operational utility under raw hourly alerting protocols**. Although the proposed representation achieved strong discrimination on the held-out cohort, direct application of the raw hourly predictions under the evaluated PhysioNet utility formulation resulted in negative normalized utility ($\text{Utility} = -1.1440$ under the prespecified validation protocol $th_{\text{val\_opt}}=0.44$; $-0.9535$ at $th=0.60$; $-0.8696$ at $th=0.78$). Patient-level decomposition indicated that missed-sepsis penalties and accumulated false-alarm penalties outweighed early-warning rewards. Thus, discrimination performance did not translate directly into positive benchmark utility under the evaluated operating protocol.

## 4.2 Why Continuous Temporal Representation Matters
In intensive care units, physiological measurements are sampled at non-uniform intervals ranging from frequent vital sign telemetry to sporadic laboratory draws. Standard sequence models often process data by assuming uniform step intervals or applying Last Observation Carried Forward (LOCF) imputation, which obscures the continuous nature of time.

Our results demonstrate that projecting elapsed time deltas ($\boldsymbol{\Delta t}$) into periodic and linear frequency embeddings via **Time2Vec** (Kazemi et al., 2019) provides substantial predictive benefits. Comparing the values-only baseline (M2) to the Time+Delta variant (M3-Time+Delta) reveals an improvement of **+0.0215 AUROC** (0.9265 $\to$ 0.9480), **+0.0350 AUPRC** (0.3540 $\to$ 0.3890), and an extension of mean lead time by **+1.0 hour** (4.2h $\to$ 5.2h). By projecting time gaps directly into the multi-head self-attention space, the model learns non-linear temporal decay profiles for individual physiological features without suffering from the step-wise recurrence bottlenecks of LSTMs or GRUs.

## 4.3 Role of Informative Missingness Patterns
A key characteristic of electronic health record data is that missingness is not missing at random (MAR); rather, test ordering decisions reflect clinical workflow and diagnostic evaluation (Che et al., 2018; Rubin et al., 2018). For instance, an increase in the frequency of arterial blood gas or lactate orders may encode signals regarding clinical concern prior to the availability of laboratory results.

Our component ablation confirms that explicitly concatenating binary observation masks ($\mathbf{m}$) provides independent diagnostic value. Adding masks to the values-only baseline (M2 $\to$ M3-Time+Mask) improved AUROC by **+0.0155** (0.9265 $\to$ 0.9420) and increased precision by **+0.0230** (0.2250 $\to$ 0.2480). When added to the Time+Delta model (M3-Time+Delta $\to$ M3-Full), missingness masks yielded an incremental **+0.0449 PPV increase** in precision (0.2650 $\to$ 0.3099) while lowering non-sepsis false alarm rates down to **0.0139 FPR/hour** (1.39% per hour at $th=0.60$). This indicates that observation patterns act as an effective precision regularizer in clinical self-attention models.

## 4.4 Operational Utility & Master Metric Reconciliation Matrix
To understand why predictive discrimination ($\text{AUROC}=0.9617$) did not yield positive challenge utility, we performed an exact patient-level forensic decomposition of the PhysioNet Utility Score across $20,000$ test patient sequences ($1,066$ septic patients, $18,934$ non-septic patients). Table 4 details the exact arithmetic decomposition across the three evaluated operating points alongside explicit distinctions between hourly recall, patient detection rate, non-sepsis FPR/h, and all-hours alarm rates.

### Table 4: Master Patient-Level Utility & Metric Reconciliation Matrix (Held-Out Test Cohort, N=20,000)
| Metric Dimension | Primary Protocol ($th=0.44$, Val Utility Opt) | Sensitivity ($th=0.60$, Balanced Fallback) | Sensitivity ($th=0.78$, Val F1 Opt) |
|---|:---:|:---:|:---:|
| **Normalized PhysioNet Utility ($U_{\text{norm}}$)** | **-1.1440** | **-0.9535** | **-0.8696** |
| **Hourly Recall (Sensitivity)** | **67.08%** | **61.03%** | **50.79%** |
| **Patient Detection Rate (Septic TPs)** | **70.4% (750 / 1,066)** | **66.0% (704 / 1,066)** | **57.5% (613 / 1,066)** |
| **Patient Missed Rate (Septic FNs)** | **29.6% (316 / 1,066)** | **34.0% (362 / 1,066)** | **42.5% (453 / 1,066)** |
| **Early Warning TP Reward** | **+233.56 pts** | **+242.89 pts** | **+222.00 pts** |
| **Missed-Sepsis FN Penalty** | **-632.00 pts (316 × -2.0)** | **-724.00 pts (362 × -2.0)** | **-906.00 pts (453 × -2.0)** |
| **Non-Sepsis FP Alarm Hours** | **14,771 hrs** | **9,816 hrs** | **4,605 hrs** |
| **Non-Sepsis FP Penalty** | **-738.55 pts** | **-490.80 pts** | **-230.25 pts** |
| **Sepsis Early FP Alarm Hours** | **1,651 hrs** | **891 hrs** | **255 hrs** |
| **Total False Alarm Penalty** | **-821.10 pts** | **-535.35 pts** | **-243.00 pts** |
| **Total Achieved Utility (Raw)** | **-1,219.54 pts** | **-1,016.46 pts** | **-927.00 pts** |
| **Total Best Possible Utility** | **+1,066.00 pts** | **+1,066.00 pts** | **+1,066.00 pts** |
| **Non-Sepsis Hourly FPR ($\text{FPR/h}_{\text{non-sepsis}}$)** | **2.10% per hour** | **1.39% per hour** | **0.65% per hour** |
| **All-Hours Hourly Alarm Rate** | **3.56% per hour** | **2.62% per hour** | **1.54% per hour** |
| **Hourly Precision (PPV)** | **25.09%** | **30.99%** | **43.90%** |
| **Hourly F1-Score** | **0.3652** | **0.4110** | **0.4710** |
| **Mean Lead Time (Hours)** | **7.7 h** | **5.7 h** | **2.9 h** |

As shown in Table 4, the exact arithmetic identity holds to numerical precision ($\text{Achieved Utility} = \text{TP Reward} + \text{FN Penalty} + \text{FP Penalty}$):
1. **Primary Protocol ($th=0.44$):** M3 detected 750 of 1,066 septic cases (70.4% patient detection rate), earning $+233.56$ points in early warning rewards. However, 316 missed cases incurred $316 \times (-2.00) = \mathbf{-632.00}$ penalty points, while $14,771$ false-alarm hours on non-septic patients plus $1,651$ early false-alarm hours incurred $\mathbf{-821.10}$ penalty points. Net achieved utility: $+233.56 - 632.00 - 821.10 = \mathbf{-1,219.54}$ points ($U_{\text{norm}} = \mathbf{-1.1440}$).
2. **Balanced Fallback ($th=0.60$):** Detecting 704 cases (66.0%) earned $+242.89$ points, but 362 missed cases ($-724.00$ points) and $9,816$ non-sepsis FP hours ($-490.80$ points) yielded $-1,016.46$ net points ($U_{\text{norm}} = \mathbf{-0.9535}$).
3. **Validation F1-Opt ($th=0.78$):** Detecting 613 cases (57.5%) earned $+222.00$ points, with false alarm hours dropping to $4,605$ hours ($-230.25$ points), but 453 missed cases ($-906.00$ points) yielded $-927.00$ net points ($U_{\text{norm}} = \mathbf{-0.8696}$).

This exact decomposition demonstrates that evaluating early warning models using raw hourly predictions hour-by-hour causes accumulated false alarms across non-septic ICU stays and missed sepsis penalties to exceed early warning rewards.

## 4.5 Architectural Complexity & MoE Routing Insights
Recent literature in clinical machine learning has seen a trend toward increasingly complex, multi-branch, and Mixture-of-Experts (MoE) architectures. To test whether such complexity is warranted, we evaluated Model M4 (Organ Hybrid with 6 PATE encoders, 198,433 parameters) and Model M5 (Multi-Hybrid with 3 disjoint branch encoders and Softmax MoE expert router, 224,713 parameters) against Model M3 (163,841 parameters).

Our empirical findings demonstrate that increasing architectural complexity did not improve discrimination or operational utility ($\text{AUROC}_{\text{M3}} = 0.9617$ vs. $\text{AUROC}_{\text{M4}} = 0.9412$ vs. $\text{AUROC}_{\text{M5}} = 0.9358$). Disjointly separating values, masks, and time into isolated branch encoders (as in M5) introduced representation friction, preventing the self-attention mechanism from learning joint cross-modal interactions early in the network. M5 generated elevated false alarm rates (5.80% FPR/h), resulting in a severely negative test utility of -2.5556. A compact Transformer backbone that projects physiological values, missingness masks, and continuous frequency temporal representations into a single unified embedding space preserves inter-feature correlations more effectively than multi-branch MoE expert partitioning.

## 4.6 Comparison with Prior Literature
Our results contextualize recent findings in clinical deep learning for sepsis alerting. While classical GBDT models like InSight (Desautels et al., 2016) and PhysioNet challenge baselines (Zabihi et al., 2020) achieved AUROCs between 0.840 and 0.880 on tabular summary windows, sequential deep learning models improved performance by modeling hourly trajectories (Scherpf et al., 2019; Zhang et al., 2021). Recent clinical Transformer adaptations (Tipirneni and Reddy, 2022; Yang et al., 2024) further demonstrated the advantages of self-attention over recurrent networks.

M3 extends this literature by demonstrating that incorporating continuous Time2Vec frequency embeddings and missingness masks directly into a causal Transformer yields state-of-the-art discrimination ($\text{AUROC} = 0.9617$, $\text{AUPRC} = 0.4231$) on the PhysioNet 2019 benchmark while maintaining calibration ($\text{ECE} = 0.0407$).

## 4.7 Practical Clinical Implications
From an operational perspective, M3 offers three practical takeaways:
1. **Calibrated Risk Scoring:** An ECE of 4.07% ensures that model output probabilities accurately reflect true physiological risk, enabling clinicians to establish trustworthy risk thresholds.
2. **Actionable Resuscitation Window:** A mean lead time of 7.7 hours (at $th=0.44$) and 5.7 hours (at $th=0.60$) aligns with clinical intervention protocols (e.g., Surviving Sepsis Campaign bundles; Evans et al., 2021), providing care teams adequate time for diagnostic workup.
3. **Necessity of Post-Processing Filters:** The negative utility scores produced by raw hourly alerting demonstrate that bedside deployment requires sequence-level post-processing filters (e.g., moving average smoothing, persistent alert requirements, or clinical hysteresis) to suppress transient false alarms.

## 4.8 Limitations and Future Directions
This study has several limitations:
1. **Retrospective Single-Benchmark Design:** Evaluation was performed retrospectively on the PhysioNet 2019 dataset across two hospital systems. Prospective multi-center clinical validation across diverse EHR databases (e.g., MIMIC-IV, eICU) is required before clinical deployment.
2. **Operational Utility Metric Constraints:** The PhysioNet Utility Score applies fixed linear penalties for false alarms and rewards for early warnings. While standardized, optimal utility weights may vary across individual institutional ICU workflows.
3. **Raw Hourly Alerting Limitation:** Predictions were evaluated hour-by-hour without sequence-level hysteresis or alert persistence filters, which accumulated false alarm penalties on non-septic patient stays.
4. **Sepsis Label Framework:** Ground truth sepsis labels rely on the Sepsis-3 challenge annotation framework. Labeling uncertainty or variations in clinical documentation timing across hospitals could affect precise onset hours.
