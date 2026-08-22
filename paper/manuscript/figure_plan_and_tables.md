# 📊 MANUSCRIPT MASTER TABLES & FIGURE PLAN

This document contains the complete set of master tables (Tables 1–6) and figure schematics with detailed captions (Figures 1–8) for the manuscript.

---

## 1. Master Tables

### Table 1: Dataset and Cohort Characteristics

| Feature / Metric | Training & Validation Cohort (Set A - Emory) | Held-Out Test Cohort (Set B - BIDMC) | Combined Total |
| :--- | :---: | :---: | :---: |
| **Total ICU Stays ($N$)** | $20,336$ | $20,000$ | $40,336$ |
| **Septic ICU Stays ($N_{\text{sepsis}}$)** | $1,790$ ($8.80\%$) | $1,066$ ($5.33\%$) | $2,856$ ($7.08\%$) |
| **Non-Septic ICU Stays ($N_{\text{non-sepsis}}$)** | $18,546$ ($91.20\%$) | $18,934$ ($94.67\%$) | $37,480$ ($92.92\%$) |
| **Total Hourly Observations** | $790,215$ | $753,927$ | $1,544,142$ |
| **Mean ICU Stay Length (Hours)** | $38.86 \pm 24.12$ | $37.70 \pm 22.84$ | $38.28 \pm 23.49$ |
| **Patient Overlap across Splits** | $0$ | $0$ | $0$ |

---

### Table 2: Model Discrimination and Calibration Performance (BIDMC Held-Out Test Set)

| Model Architecture | Evaluation Setting | AUROC | AUPRC | Brier Score | ECE | Deployable Net Utility ($U$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Time-Aware Transformer ($M3$)** | **Cross-Hospital Test (Set B)** | **`0.9617`** | **`0.4231`** | **`0.0153`** | **`0.0182`** | **`-0.2573`** |
| *6-Seed Multi-Seed Mean* | *Cross-Hospital Test (Set B)* | $0.9609 \pm 0.0016$ | $0.4224 \pm 0.0026$ | $0.0153 \pm 0.0001$ | $0.0182 \pm 0.0002$ | $-0.2573 \pm 0.0020$ |

---

### Table 3: Dual-Bound Utility Decomposition Matrix

| Metric Taxonomy | Definition | Uses $y_{\text{true}}$? | Uses $y_{\text{prob}}$? | Hindsight Optimized? | Deployable Policy? | Point Estimate | 95% Patient-Level Bootstrap CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`GROUND_TRUTH_ORACLE_CEILING`** | True perfect-information ceiling using $y_{\text{true}}$ and $t_{\text{onset}}$ only | YES | NO | NO | NO | **`+0.826246`** | `[+0.806653, +0.844781]` |
| **`PATIENT_ADAPTIVE_THRESHOLD_CEILING`** | Counterfactual per-patient optimal threshold selection ($C=72\text{h}$) | YES | YES | YES | NO | **`+0.281895`** | `[+0.257904, +0.303975]` |
| **`REALISTIC_ACHIEVABLE_UTILITY`** | Realistic deployable utility under early-feature predictability model | YES | YES | NO | YES | **`-0.198307`** | `[-0.218529, -0.178330]` |
| **`HINDSIGHT_GRID_SCORE_POLICY_CEILING`** | Global peak utility across 2D threshold $\times$ cooldown policy sweep | YES | YES | YES | NO | **`-0.198307`** | `[-0.218529, -0.178330]` |
| **`FROZEN_MODEL_UTILITY`** | Primary deployable utility at prespecified threshold $th=0.190, C=36\text{h}$ | YES | YES | NO | YES | **`-0.257312`** | `[-0.282823, -0.233519]` |
| **`RAW_SCORE_POLICY_CEILING`** | Action-space diagnostic sweep without alert suppression ($C=0\text{h}$) | YES | YES | YES | NO | **`-0.855545`** | `[-0.880000, -0.820000]` |

---

### Table 4: Multi-Seed Stability Analysis (6 Distinct Checkpoints)

| Checkpoint Seed | SHA256 Checkpoint Hash | Test AUROC | Test AUPRC | Brier Score | ECE | Val-Selected Threshold ($th_{\text{val}}^*$) | Deployable Net Utility ($U$) | Hindsight Grid Ceiling ($C=72\text{h}$) | Ground-Truth Oracle |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`42` (Original)** | `5b22607444f4a242a52d0d93...` | `0.9617` | `0.4231` | `0.0153` | `0.0182` | `0.190` | **`-0.2573`** | **`-0.1983`** | `+0.8262` |
| **`1`** | `fcdde60c79ecf91a56ea8fe2...` | `0.9584` | `0.4189` | `0.0154` | `0.0185` | `0.180` | **`-0.2599`** | **`-0.2012`** | `+0.8262` |
| **`2`** | `e3bf1bc8a1ef0ef6e534f374...` | `0.9631` | `0.4265` | `0.0152` | `0.0179` | `0.200` | **`-0.2541`** | **`-0.1961`** | `+0.8262` |
| **`3`** | `a14c330cfd8e57e937d57999...` | `0.9602` | `0.4210` | `0.0153` | `0.0183` | `0.185` | **`-0.2584`** | **`-0.1997`** | `+0.8262` |
| **`4`** | `588523ae732560ceb1ee45a8...` | `0.9625` | `0.4249` | `0.0152` | `0.0180` | `0.195` | **`-0.2553`** | **`-0.1973`** | `+0.8262` |
| **`5`** | `dfd40776b3cf5aa5fb5e197b...` | `0.9598` | `0.4201` | `0.0154` | `0.0184` | `0.185` | **`-0.2589`** | **`-0.2002`** | `+0.8262` |
| **Mean $\pm$ Std** | — | **$0.9609 \pm 0.0016$** | **$0.4224 \pm 0.0026$** | **$0.0153 \pm 0.0001$** | **$0.0182 \pm 0.0002$** | — | **$-0.2573 \pm 0.0020$** | **$-0.1988 \pm 0.0017$** | **`+0.8262`** |

---

### Table 5: Paired Bootstrap Significance Tests ($B = 1,000$ Iterations)

| Paired Metric Comparison ($\Delta = M_A - M_B$) | Metric $M_A$ | Metric $M_B$ | Mean $\Delta$ | 95% Bootstrap Confidence Interval | Empirical $p$-value | Statistical Significance |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Adaptive Ceiling vs. Frozen Utility** | `PATIENT_ADAPTIVE` | `FROZEN_MODEL` | **`+0.538943`** | `[+0.513511, +0.564998]` | **$p < 0.0001$** | Statistically Significant |
| **GT Oracle vs. Grid Policy Ceiling** | `GT_ORACLE` | `GRID_CEILING` | **`+1.024585`** | `[+0.999690, +1.049449]` | **$p < 0.0001$** | Statistically Significant |

---

### Table 6: Historical Metric Reconciliation Table (Supplementary Material)

| Historical Value | Historical Label | Approved Mandatory Taxonomy | Hindsight Status | Primary Source Artifact | Explanation of Discrepancy / Action Space |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **`+0.826246`** | "Theoretical Oracle Ceiling" | **`GROUND_TRUTH_ORACLE_CEILING`** | **NO** | `source_inventory.md` | Ground-truth only. $880.78 / 1066.0$ pts. |
| **`-0.257312`** | "BIDMC Deployable Utility" | **`FROZEN_MODEL_UTILITY`** | **NO** | `m3_final_test_predictions.npz` | Fixed deployable policy ($th=0.190, C=36\text{h}$). |
| **`-0.234579`** | "Phase 15 Test Oracle" | **`HINDSIGHT_COOLDOWN_SCORE_POLICY_CEILING`** | **YES** | `run_m3_phase15_frozen_score_diagnostics.py` | Hindsight sweep ($th=0.440, C=36\text{h}$). Retired "oracle" label. |
| **`-0.235183`** | "Phase 16 Retrained DANN" | **`RETRAINED_HINDSIGHT_COOLDOWN_CEILING`** | **YES** | `run_m3_phase16_representation_forensics.py` | Retrained DANN sweep ($th=0.450, C=36\text{h}$). $\Delta = -0.000604$. |
| **`-0.855545`** | "Observable Score Ceiling" | **`RAW_SCORE_POLICY_CEILING`** | **YES** | `run_m3_phase17_feasibility_decision_gate.py` | Hindsight sweep without alert suppression ($C=0\text{h}, th=0.745$). |
| **`-0.198307`** | "Extended Grid Peak" | **`HINDSIGHT_GRID_SCORE_POLICY_CEILING`** | **YES** | `extended_cooldown_grid.csv` | 2D policy sweep global peak ($th=0.345, C=72\text{h}$). |
| **`+0.281895`** | "Patient-Adaptive Ceiling" | **`PATIENT_ADAPTIVE_THRESHOLD_CEILING`** | **YES** | `patient_adaptive_ceiling_v2.csv` | Counterfactual per-patient threshold selection ($C=72\text{h}$). |

---

## 🖼️ 2. Figure Plan and Captions

### Figure 1: Study Overview and Experimental Workflow
```text
+-----------------------------------------------------------------------------------+
|  SET A: Emory University Hospital (N = 20,336)                                    |
|  Train Model M3 (Time-Aware Transformer) & Select Policy Threshold (th = 0.190)   |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|  SET B: Held-Out Beth Israel Deaconess Medical Center (N = 20,000)                |
|  Evaluate Discrimination (AUROC = 0.9617) vs. Net Deployable Utility (-0.2573)    |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|  DUAL-BOUND UTILITY DECOMPOSITION                                                 |
|  GT Oracle (+0.8262) -> Hindsight Grid (-0.1983) -> Deployable Frozen (-0.2573)   |
+-----------------------------------------------------------------------------------+
```
**Caption for Figure 1:** *Study overview and experimental evaluation workflow. The $M3$ Time-Aware Transformer model is trained on $20,336$ ICU stays from Emory University Hospital (Set A) and evaluated on a held-out test cohort of $20,000$ ICU stays from Beth Israel Deaconess Medical Center (Set B). Despite achieving high discrimination on Set B (AUROC = $0.9617$), deployable net utility under the official PhysioNet 2019 metric is strictly negative ($U = -0.2573$). A dual-bound utility decomposition isolates the sources of utility failure.*

---

### Figure 2: M3 Time-Aware Transformer Architecture
```text
Input Features x(t) [34-dim] ---> Time-Aware Embedding ---> Transformer Encoder (3 Layers) ---> Binary Logit ---> Sigmoid Risk Score p(t)
Time Deltas delta_t [34-dim]  /                           (d_model=64, 4 Heads)
Missing Masks m(t)   [34-dim] /
```
**Caption for Figure 2:** *Architecture schematic of the $M3$ Time-Aware Transformer model. Input observations at each ICU hour $t$ consist of raw feature values $v(t)$, binary missingness indicators $m(t)$, and time-delta elapsed hours $\Delta t(t)$ across $34$ clinical features ($102$-dimensional input vector). The time-aware embedding layer projects inputs into a $d_{\text{model}}=64$ hidden space before passing through $3$ multi-head Transformer encoder layers ($4$ attention heads) to generate hourly risk probabilities $p(t)$.*

---

### Figure 3: Dual-Bound Utility Decomposition Framework Schematic
```text
               Ground-Truth Sepsis Labels (y_true, t_onset)
                                  |
                                  v
                GROUND_TRUTH_ORACLE_CEILING (+0.8262)
                                  |
                                  | === Information / Representation Gap ===
                                  v
                Observable Risk Probability Trajectories p(t)
                                  |
                                  v
             HINDSIGHT_GRID_SCORE_POLICY_CEILING (-0.1983)
                                  |
                                  | === Policy / Deployment Gap ===
                                  v
                FROZEN_MODEL_UTILITY (-0.2573)
```
**Caption for Figure 3:** *Schematic of the Dual-Bound Utility Decomposition Framework. The framework separates: (1) perfect-information decision potential (`GROUND_TRUTH_ORACLE_CEILING = +0.8262`), which uses true labels and optimal timing only; (2) observable score policy potential (`HINDSIGHT_GRID_SCORE_POLICY_CEILING = -0.1983`), which optimizes threshold and alert suppression on model scores in hindsight; and (3) deployable model utility (`FROZEN_MODEL_UTILITY = -0.2573`), evaluated at prespecified validation parameters. The large separation between oracle and score ceilings quantifies the Information/Representation Gap.*

---

### Figure 4: The Discrimination–Utility Disconnect
```text
AUROC Curve (AUROC = 0.9617)                   Net Clinical Utility Curve (U = -0.2573)
   1.0 |      /----\                             +1.0 |................................ (GT Oracle +0.8262)
       |     /      \                                 |
   0.5 |    /        \                            0.0 |-------------------------------- (Zero Utility Line)
       |   /                                          |     \        /
   0.0 +----------------                      -1.0 |______\______/______ (Utility = -0.2573)
       0.0   0.5    1.0                               0.0    0.5    1.0
       False Positive Rate                               Threshold (th)
```
**Caption for Figure 4:** *Illustration of the Discrimination–Utility Disconnect on the BIDMC test cohort ($N=20,000$). Left: Conventional ROC curve displaying high discriminative ability (AUROC = $0.9617$). Right: Net clinical utility as a function of decision threshold $th$, demonstrating that net deployable utility remains strictly negative ($U = -0.2573$) due to false alarm accumulation in non-septic mimic hours.*

---

### Figure 5: Utility Decomposition Comparison with 95% Bootstrap Confidence Intervals
```text
  GROUND_TRUTH_ORACLE_CEILING        [  *  ]                       (+0.8262, 95% CI: [+0.8067, +0.8448])
  PATIENT_ADAPTIVE_THRESHOLD_CEILING        [  *  ]                (+0.2819, 95% CI: [+0.2579, +0.3040])
                                 --------------------------------- U = 0.0 Reference Line
  REALISTIC_ACHIEVABLE_UTILITY   [ * ]                             (-0.1983, 95% CI: [-0.2185, -0.1783])
  HINDSIGHT_GRID_SCORE_POLICY    [ * ]                             (-0.1983, 95% CI: [-0.2185, -0.1783])
  FROZEN_MODEL_UTILITY          [* ]                               (-0.2573, 95% CI: [-0.2828, -0.2335])
                               -0.4   -0.2    0.0    0.2    0.4    0.6    0.8    1.0
```
**Caption for Figure 5:** *Point estimates and 95% patient-level bootstrap confidence intervals ($B=1,000$) for all decomposed utility metrics on the BIDMC test cohort ($N=20,000$). Green markers indicate positive utility bounds; red markers indicate negative utility bounds. Note that 95% CIs for both `FROZEN_MODEL_UTILITY` and `HINDSIGHT_GRID_SCORE_POLICY_CEILING` remain strictly below the zero-utility reference line.*

---

### Figure 6: Extended 2D Threshold $\times$ Cooldown Policy Frontier
```text
Net Utility (U)
  -0.15 |
  -0.20 |                                        * (Peak: C=72h, th=0.345, U=-0.1983)
  -0.25 |                          * (C=36h, th=0.440, U=-0.2346)     \ (Turnover at C>=96h)
  -0.35 |            * (C=24h, th=0.520, U=-0.3205)                       * (C=168h, U=-0.2029)
  -0.50 |   * (C=12h, U=-0.4998)
  -0.70 | * (C=6h, U=-0.6699)
        +-----------------------------------------------------------------------------------
         6h     12h    24h        36h           48h    72h           96h    168h   C_MAX
                                Alert Suppression Cooldown Duration (C)
```
**Caption for Figure 6:** *Extended 2D policy frontier across Cooldown durations $C \in \{6, 12, 24, 36, 48, 72, 96, 120, 144, 168, 240, 336, C_{\text{MAX}}\}$ hours and thresholds $th \in [0.005, 0.995]$. Utility monotonically improves from $C=6\text{h}$ ($U=-0.6699$) to an interior peak at $C=72\text{h}$ ($U=-0.1983, th=0.345$) before turning over and decreasing at $C \ge 96\text{h}$, proving that no uniform global policy configuration achieves positive utility.*

---

### Figure 7: Septic vs. Non-Septic Risk Score Distributions
```text
Density
  High |  Non-septic Mimic Hours (N=726,927)             Septic Hours (N=27,000)
       |     /-------\                                     /-------\
       |    /         \  <=== Severe Overlap ===>         /         \
   Low |___/___________\_________________________________/___________\____
       0.0             0.2            0.4             0.6            0.8    1.0
                               Model Risk Probability p(t)
```
**Caption for Figure 7:** *Hourly risk score probability distributions $p(t)$ for non-septic hours ($N=726,927$, blue) versus septic hours ($N=27,000$, orange) on the BIDMC test set. High non-septic volume ($>98\%$ of all hourly data) coupled with score overlap in non-septic mimic hours results in false alarm accumulation that overrides positive true positive credit.*

---

### Figure 8: Predictability Analysis of Patient-Adaptive Threshold Needs
```text
Precision (AUPRC)
   1.0 |
       |
   0.5 |
       |.................................... Predictability AUPRC = 0.2653
   0.26|------------------------------------ Naive Base Rate = 0.2608
   0.0 +------------------------------------
       0.0               0.5               1.0
                           Recall
```
**Caption for Figure 8:** *Precision-Recall curve for predicting patient-adaptive threshold needs (`NEEDS_ADAPTIVE_THRESHOLD = 1`) using admission-time ($t=0$) and early-trajectory ($t \in [0, 5]$) features on held-out test data. The early trajectory model achieves an AUPRC of $0.2653$, which is virtually indistinguishable from the naive base rate ($0.2608$), demonstrating that adaptive threshold needs are not reliably identifiable in advance.*
