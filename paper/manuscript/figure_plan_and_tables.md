# 📊 REVISED MANUSCRIPT MASTER TABLES & FIGURE PLAN

This document contains the complete set of revised master tables (Tables 1–8), cohort provenance table, TRIPOD+AI checklist, and figure schematics/captions (Figures 1–6) for the manuscript.

---

## 1. Master Tables

### Table 1: Dataset Cohort Provenance & Lineage Matrix

| Partition / Cohort Name | Official PhysioNet 2019 Source Hospital | Patient Stays ($N$) | Septic ICU Stays ($N_{\text{sepsis}}$) | Non-Septic ICU Stays | Total Hourly Records | Split Manifest / Prediction Artifact |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Set A (Development)** | **Beth Israel Deaconess Medical Center (BIDMC)** | $20,336$ | $1,790$ ($8.80\%$) | $18,546$ ($91.20\%$) | $790,215$ | `data/splits/train_ids.json` & `val_ids.json` |
| **Set B (Cross-Hospital Test)** | **Emory University Hospital** | $20,000$ | $1,066$ ($5.33\%$) | $18,934$ ($94.67\%$) | $753,927$ | `data/splits/test_ids.json` & `m3_final_test_predictions.npz` |
| **Combined Benchmark Total** | **Two Health Systems** | $40,336$ | $2,856$ ($7.08\%$) | $37,480$ ($92.92\%$) | $1,544,142$ | Cryptographic SHA256 Manifest Verified |

---

### Table 2: Extended Cross-Hospital Benchmark Comparison (Set B Emory Test Set, N=20,000)

| Model ID | Architecture Name & Class | AUROC | AUPRC | Brier Score | ECE | Deployable Net Utility ($U$) | Parameter Count | Primary Modeling Focus |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`M1`** | XGBoost Baseline | `0.8842` | `0.2851` | `0.0241` | `0.0382` | `-0.4812` | 150K | Classical Gradient Boosting |
| **`M2`** | Plain Transformer (Values Only) | `0.9265` | `0.3412` | `0.0189` | `0.0245` | `-0.3894` | 180K | Standard Self-Attention |
| **`GRU-D`** | GRU-D (Che et al., 2018) | `0.9415` | `0.3780` | `0.0171` | `0.0210` | `-0.3120` | 145K | Missingness-Aware Recurrent NN |
| **`TCN`** | Temporal Convolutional Network | `0.9380` | `0.3650` | `0.0175` | `0.0225` | `-0.3350` | 160K | Temporal 1D Convolutions |
| **`PhysioNet`** | PhysioNet 2019 Challenge Baseline | `0.8420` | `0.2150` | `0.0310` | `0.0520` | `-0.5820` | Rule-based | Heuristic Persistence Model |
| **`M3`** | **Time-Aware Transformer (Full Triplet)** | **`0.9617`** | **`0.4231`** | **`0.0153`** | **`0.0182`** | **`-0.2573`** | 185K | **Compact Time-Aware Transformer** |
| **`M4`** | Organ-Aware Hybrid Architecture | `0.9582` | `0.4150` | `0.0158` | `0.0195` | `-0.2641` | 320K | Dual-Branch Organ Structure |
| **`M5`** | Multi-Hybrid / MoE Architecture | `0.9591` | `0.4182` | `0.0156` | `0.0190` | `-0.2610` | 450K | Mixture-of-Experts Routing |

---

### Table 3: Factorial M3 Ablation Matrix Across 5 Random Seeds

| Factorial Variant | Physiological Values ($v$) | Missingness Mask ($m$) | Elapsed Time Delta ($\Delta t$) | Test AUROC (Mean $\pm$ Std) | Test AUPRC | Deployable Net Utility | Factorial Effect Estimate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Values Only (Baseline)** | YES | NO | NO | $0.9265 \pm 0.0022$ | $0.3412$ | $-0.3894$ | Baseline Reference |
| **Mask Contribution** | YES | YES | NO | $0.9420 \pm 0.0019$ | $0.3751$ | $-0.3150$ | **Main Effect of Mask ($m$): $+0.0155$ AUROC** |
| **Time Delta Contribution** | YES | NO | YES | $0.9480 \pm 0.0018$ | $0.3895$ | $-0.2980$ | **Main Effect of Time ($\Delta t$): $+0.0215$ AUROC** |
| **Full M3 (Interaction)** | YES | YES | YES | **$0.9617 \pm 0.0016$** | **$0.4231$** | **$-0.2573$** | **Interaction ($m \times \Delta t$): $+0.0017$ AUROC** |

---

### Table 4: Operational Workload & Alert Burden Metrics (Set B Emory Test Set)

| Workload Metric | Metric Value | Clinical / Operational Interpretation |
| :--- | :---: | :--- |
| **Total Alerts Issued** | $5,337$ alerts | $1,004$ True Sepsis Alerts, $4,333$ Non-Sepsis False Alerts |
| **Total ICU Patient-Days** | $31,413.6$ patient-days | $753,927$ total hourly observations / $24.0$ hours |
| **Alert Frequency** | **$16.99$ alerts / 100 patient-days** | Operational clinical alert rate |
| **Alerts per Patient** | **$0.267$ alerts / patient** | Average alert burden per ICU stay |
| **False Alerts per Non-Septic Patient** | **$0.229$ false alerts / patient** | Nuisance alarm burden on non-septic patients |
| **Alert Positive Predictive Value (PPV)** | **$18.81\%$** | $1,004$ True Positive Alerts / $5,337$ Total Alerts Issued |
| **Percentage of Patients Alerted** | **$25.86\%$** | $5,172$ out of $20,000$ ICU stays triggered an alert |

---

### Table 5: Refined Dual-Bound Utility Decomposition & Patient-Level Bootstrap CIs ($B=1,000$)

| Metric Taxonomy | Point Estimate | 95% Bootstrap CI | Uses $y_{\text{true}}$? | Uses $y_{\text{prob}}$? | Deployable Status | Refined Observability & Infeasibility Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`GROUND_TRUTH_ORACLE_CEILING`** | **`+0.826246`** | `[+0.806653, +0.844781]` | YES | NO | **Infeasible Upper Bound** | Label-informed upper bound ($880.78 / 1066.0$ pts) |
| **`PATIENT_ADAPTIVE_THRESHOLD_CEILING`** | **`+0.281895`** | `[+0.257904, +0.303975]` | YES | YES | **Infeasible Upper Bound** | Counterfactual per-patient threshold selection ($C=72\text{h}$) |
| **`REALISTIC_ACHIEVABLE_UTILITY`** | **`-0.198307`** | `[-0.218529, -0.178330]` | YES | YES | **Deployable Policy** | Deployable utility under locked predictability model ($AUPRC=0.2653$) |
| **`HINDSIGHT_GRID_SCORE_POLICY_CEILING`** | **`-0.198307`** | `[-0.218529, -0.178330]` | YES | YES | Hindsight Sweep | Global peak utility across 2D threshold $\times$ cooldown sweep ($C=72\text{h}$) |
| **`FROZEN_MODEL_UTILITY`** | **`-0.257312`** | `[-0.282823, -0.233519]` | YES | YES | **Primary Deployable Policy** | Fixed deployable policy at prespecified threshold ($th=0.190, C=36\text{h}$) |
| **`ORACLE_TO_GLOBAL_POLICY_UTILITY_GAP`** | **`+1.024585`** ($p < 0.0001$) | `[+0.999690, +1.049449]` | YES | YES | Composite Diagnostic Gap | Observed gap between oracle bound and global score policy ceiling |

---

### Table 6: Paired Bootstrap Significance Comparisons ($B=1,000$ Iterations)

| Paired Model / Policy Comparison | Model / Policy A | Model / Policy B | Mean Difference ($\Delta$) | 95% Bootstrap Confidence Interval | Statistical Significance |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **M3 vs. XGBoost (M1)** | $M3$ (Time-Aware) | $M1$ (XGBoost) | **$+0.0775$** (AUROC) | `[+0.0712, +0.0838]` | $p < 0.0001$ |
| **M3 vs. Plain Transformer (M2)** | $M3$ (Time-Aware) | $M2$ (Plain) | **$+0.0352$** (AUROC) | `[+0.0310, +0.0394]` | $p < 0.0001$ |
| **M3 vs. GRU-D** | $M3$ (Time-Aware) | GRU-D | **$+0.0202$** (AUROC) | `[+0.0158, +0.0246]` | $p < 0.0001$ |
| **M3 vs. TCN** | $M3$ (Time-Aware) | TCN | **$+0.0237$** (AUROC) | `[+0.0191, +0.0283]` | $p < 0.0001$ |
| **M3 vs. Organ-Aware (M4)** | $M3$ (Time-Aware) | $M4$ (Organ-Aware) | **$+0.0035$** (AUROC) | `[+0.0008, +0.0062]` | $p = 0.012$ |
| **M3 vs. Multi-Hybrid (M5)** | $M3$ (Time-Aware) | $M5$ (MoE) | **$+0.0026$** (AUROC) | `[-0.0002, +0.0054]` | $p = 0.068$ (N.S.) |

---

### Table 7: TRIPOD+AI Reporting Checklist Evaluation

| TRIPOD+AI Item | Description | Compliance Status | Section Location in Paper |
| :--- | :--- | :---: | :--- |
| **Title & Abstract** | Identify study as ML prediction model development & cross-site validation | **COMPLIANT** | Title & Abstract |
| **Source of Data** | Describe study design, health system sources, and data collection dates | **COMPLIANT** | Methods Section 3.1 & Table 1 |
| **Participants** | State eligibility criteria, exclusion rules, and sample flow | **COMPLIANT** | Methods Section 3.1 & 3.2 |
| **Predictors** | Define all physiological, lab, and demographic predictors used | **COMPLIANT** | Methods Section 3.3 |
| **Outcome** | Define Sepsis-3 outcome criteria, onset window, and timing determination | **COMPLIANT** | Methods Section 3.1 & 3.9 |
| **Sample Size** | Report total patient counts, septic events, and hourly records | **COMPLIANT** | Methods Section 3.1 & Table 1 |
| **Model Specification** | Provide detailed neural network, embedding, and loss function specifications | **COMPLIANT** | Methods Section 3.4 & 3.5 |
| **External Validation** | Evaluate model on an independent external hospital cohort (Set B Emory) | **COMPLIANT** | Results Section 4.1 & 4.3 |
| **Calibration & Discrimination**| Report AUROC, AUPRC, Brier score, ECE, and reliability metrics | **COMPLIANT** | Results Section 4.1 & Table 2 |
| **Clinical Utility & Risk** | Evaluate deployable net utility, decision ceilings, and workload alert burden | **COMPLIANT** | Results Section 4.4, 4.5 & Table 4 |

---

## 🖼️ 2. Figure Plan and Captions (6 Focused Main-Text Figures)

### Figure 1: Study Design & Data Provenance Schematic
```text
SET A: BIDMC Development Cohort (N=20,336 Patients)       SET B: Emory External Test Cohort (N=20,000 Patients)
Train Models (M1-M5) & Select Policy (th=0.190, C=36h) --> Locked Evaluation (AUROC=0.9617, Net Utility=-0.2573)
```
**Caption for Figure 1:** *Study design and cohort provenance flow. Models are trained and validated on $20,336$ ICU stays from Beth Israel Deaconess Medical Center (Set A) and evaluated on a held-out test cohort of $20,000$ ICU stays from Emory University Hospital (Set B). Despite achieving high discrimination on Set B (AUROC = $0.9617$), deployable net utility under the official PhysioNet 2019 metric is strictly negative ($U = -0.2573$).*

---

### Figure 2: Model Discrimination Comparison Across Model Family ($M1$–$M5$ & GRU-D)
```text
AUROC Curves                                    Precision-Recall Curves
  1.0 |    /--- M3 (0.9617)                       1.0 |
      |   /---- GRU-D (0.9415)                        |    /--- M3 (0.4231)
  0.5 |  /----- M2 (0.9265)                       0.5 |   /---- GRU-D (0.3780)
      | /------ M1 (0.8842)                           |  /----- M2 (0.3412)
  0.0 +----------------                           0.0 +----------------
      0.0       0.5       1.0                         0.0       0.5       1.0
```
**Caption for Figure 2:** *ROC (left) and PR (right) curves on the held-out Emory test cohort ($N=20,000$). The Time-Aware Transformer ($M3$, AUROC = $0.9617$, AUPRC = $0.4231$) outperforms XGBoost ($M1$), Plain Transformers ($M2$), GRU-D ($0.9415$), and TCN ($0.9380$).*

---

### Figure 3: Factorial Ablation Analysis of $M3$ Representation Encodings
```text
AUROC
 0.9617 |                                                [ Full M3 (v, m, delta_t) ]
 0.9480 |                        [ Time Delta (v, delta_t) ]
 0.9420 |    [ Mask (v, m) ]
 0.9265 | [ Values (v) ]
        +----------------------------------------------------------------------------
```
**Caption for Figure 3:** *Factorial ablation analysis evaluating the contribution of missingness masks ($m$) and time deltas ($\Delta t$). Main effect of missingness mask = $+0.0155$ AUROC; main effect of time delta = $+0.0215$ AUROC; interaction effect = $+0.0017$ AUROC.*

---

### Figure 4: The Discrimination–Utility Disconnect & Operational Workload Frontier
```text
Deployable Net Utility (U)                      Alert Burden (Alerts / 100 Patient-Days)
   0.0 |----------------------------               30 |
       |                                           20 |              * 16.99 Alerts/100 Days
  -0.2 |........ * Peak (C=72h, U=-0.1983)         10 |
       |       / \                                  0 +----------------------------------
  -0.4 |______/*__\_________________                 0.0      0.2      0.4      0.6      0.8
      0.0    0.2   0.4   0.6   0.8                             Threshold (th)
```
**Caption for Figure 4:** *Operational evaluation on the Emory test cohort. Left: Net utility across thresholds and alert suppression cooldowns, demonstrating that deployable utility remains negative across all global score policies. Right: Operational alert burden ($16.99$ alerts per $100$ patient-days at prespecified threshold $th=0.190$).*

---

### Figure 5: Dual-Bound Utility Decomposition Ladder & 95% Bootstrap CIs
```text
  GROUND_TRUTH_ORACLE_CEILING (Infeasible)    [  *  ]               (+0.8262, 95% CI: [+0.8067, +0.8448])
  PATIENT_ADAPTIVE_THRESHOLD_CEILING                 [  *  ]        (+0.2819, 95% CI: [+0.2579, +0.3040])
                                         -------------------------- U = 0.0 Reference Line
  REALISTIC_ACHIEVABLE_UTILITY           [ * ]                      (-0.1983, 95% CI: [-0.2185, -0.1783])
  HINDSIGHT_GRID_SCORE_POLICY            [ * ]                      (-0.1983, 95% CI: [-0.2185, -0.1783])
  FROZEN_MODEL_UTILITY (Deployable)     [* ]                        (-0.2573, 95% CI: [-0.2828, -0.2335])
```
**Caption for Figure 5:** *Point estimates and 95% patient-level bootstrap confidence intervals ($B=1,000$) for decomposed utility metrics on Emory test data. The Oracle-to-Global-Policy Utility Gap ($\Delta = +1.0246, p < 0.0001$) quantifies the observed separation between the infeasible label-informed upper bound and deployable global score policies.*

---

### Figure 6: Leakage-Safe Predictability Analysis of Patient-Adaptive Threshold Needs
```text
Precision (AUPRC)
   1.0 |
   0.5 |
  0.265|.................................... Test AUPRC = 0.2653 (Locked Model)
  0.260|------------------------------------ Naive Prevalence Base Rate = 0.2608
   0.0 +------------------------------------
       0.0               0.5               1.0
                           Recall
```
**Caption for Figure 6:** *Precision-Recall curve of the leakage-safe predictability model trained on Set A (BIDMC) and evaluated once on Set B (Emory). Test AUPRC ($0.2653$) is virtually indistinguishable from random baseline ($0.2608$), demonstrating that adaptive threshold requirements are not predictable from early features.*
