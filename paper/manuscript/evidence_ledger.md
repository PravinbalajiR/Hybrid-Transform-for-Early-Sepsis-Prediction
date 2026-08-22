# 📓 REVISED INTERNAL EVIDENCE LEDGER & COHORT PROVENANCE AUDIT

This Evidence Ledger maps every numerical claim, baseline architecture result, operational workload metric, and cohort provenance detail in the revised publication paper to authoritative source artifacts.

---

## 1. Cohort Provenance & Lineage Audit

| Dataset Partition | Official PhysioNet 2019 Source Hospital | Patient Count ($N$) | Sepsis Count ($N_{\text{sepsis}}$) | Non-Sepsis Count | Total Hourly Observations | File Identifier / Split Manifest |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Set A (Development)** | **Beth Israel Deaconess Medical Center (BIDMC / Hospital A)** | $20,336$ | $1,790$ ($8.80\%$) | $18,546$ ($91.20\%$) | $790,215$ | `data/splits/train_ids.json` & `val_ids.json` |
| **Set B (Cross-Hospital Test)** | **Emory University Hospital (Hospital B)** | $20,000$ | $1,066$ ($5.33\%$) | $18,934$ ($94.67\%$) | $753,927$ | `data/splits/test_ids.json` & `m3_final_test_predictions.npz` |

---

## 2. Master Model & Baseline Evidence Matrix (Set B Held-Out Test Set, N=20,000)

| Model ID | Architecture Name & Class | AUROC | AUPRC | Brier Score | ECE | Deployable Net Utility | Parameter Count | Source Artifact |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`M1`** | XGBoost Baseline (Summary Statistics) | `0.8842` | `0.2851` | `0.0241` | `0.0382` | `-0.4812` | 150K | `results/revised_publication/extended_baselines_summary.csv` |
| **`M2`** | Plain Transformer (Values Only) | `0.9265` | `0.3412` | `0.0189` | `0.0245` | `-0.3894` | 180K | `results/revised_publication/extended_baselines_summary.csv` |
| **`GRU-D`** | GRU-D (Che et al., 2018 Recurrent NN) | `0.9415` | `0.3780` | `0.0171` | `0.0210` | `-0.3120` | 145K | `results/revised_publication/extended_baselines_summary.csv` |
| **`TCN`** | Temporal Convolutional Network | `0.9380` | `0.3650` | `0.0175` | `0.0225` | `-0.3350` | 160K | `results/revised_publication/extended_baselines_summary.csv` |
| **`PhysioNet`** | PhysioNet 2019 Challenge Baseline | `0.8420` | `0.2150` | `0.0310` | `0.0520` | `-0.5820` | Rule-based | `results/revised_publication/extended_baselines_summary.csv` |
| **`M3`** | **Time-Aware Transformer (Full Triplet)** | **`0.9617`** | **`0.4231`** | **`0.0153`** | **`0.0182`** | **`-0.2573`** | 185K | `results/revised_publication/extended_baselines_summary.csv` |
| **`M4`** | Organ-Aware Hybrid Architecture | `0.9582` | `0.4150` | `0.0158` | `0.0195` | `-0.2641` | 320K | `results/revised_publication/extended_baselines_summary.csv` |
| **`M5`** | Multi-Hybrid / MoE Architecture | `0.9591` | `0.4182` | `0.0156` | `0.0190` | `-0.2610` | 450K | `results/revised_publication/extended_baselines_summary.csv` |

---

## 3. Factorial M3 Ablation Evidence Matrix

| Factorial Variant | Values ($v$) | Mask ($m$) | Delta ($\Delta t$) | AUROC Mean $\pm$ Std | AUPRC Mean | Utility Mean | Factorial Effect Type & Estimate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Values Only (Baseline)** | YES | NO | NO | $0.9265 \pm 0.0022$ | $0.3412$ | $-0.3894$ | Baseline Reference |
| **Mask Contribution** | YES | YES | NO | $0.9420 \pm 0.0019$ | $0.3751$ | $-0.3150$ | **Main Effect of Missingness Mask ($m$): $+0.0155$ AUROC** |
| **Time Delta Contribution** | YES | NO | YES | $0.9480 \pm 0.0018$ | $0.3895$ | $-0.2980$ | **Main Effect of Time Delta ($\Delta t$): $+0.0215$ AUROC** |
| **Full M3 (Interaction)** | YES | YES | YES | **$0.9617 \pm 0.0016$** | **$0.4231$** | **$-0.2573$** | **Interaction Effect ($m \times \Delta t$): $+0.0017$ AUROC** |

---

## 4. Operational Workload & Alert Burden Metrics (Set B Test Set)

| Workload Metric | Metric Value | Clinical / Operational Interpretation |
| :--- | :---: | :--- |
| **Total Alerts Issued** | $3,418$ alerts | $941$ True Sepsis Alerts, $2,477$ Non-Sepsis False Alerts |
| **Total ICU Patient-Days** | $31,413.6$ patient-days | $753,927$ total hourly observations / $24.0$ hours |
| **Alert Frequency** | **$10.88$ alerts / 100 patient-days** | Operational clinical alert rate |
| **Alerts per Patient** | **$0.171$ alerts / patient** | Average alert burden per ICU stay |
| **False Alerts per Non-Septic Patient** | **$0.131$ false alerts / patient** | Nuisance alarm burden on non-septic patients |
| **Alert Positive Predictive Value (PPV)** | **$27.53\%$** | $941$ True Positive Alerts / $3,418$ Total Alerts Issued |
| **Percentage of Patients Alerted** | **$11.24\%$** | $2,248$ out of $20,000$ ICU stays triggered an alert |

---

## 5. Renamed Utility Decomposition & Bootstrap Uncertainty Matrix ($B=1,000$)

| Metric Taxonomy | Exact Point Estimate | 95% Patient-Level Bootstrap CI | Uses $y_{\text{true}}$? | Uses $y_{\text{prob}}$? | Deployable Status | Refined Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`GROUND_TRUTH_ORACLE_CEILING`** | **`+0.826246`** | `[+0.806653, +0.844781]` | YES | NO | **Infeasible Upper Bound** | Infeasible label-informed ceiling ($880.78 / 1066.0$ pts) |
| **`PATIENT_ADAPTIVE_THRESHOLD_CEILING`** | **`+0.281895`** | `[+0.257904, +0.303975]` | YES | YES | **Infeasible Upper Bound** | Counterfactual per-patient threshold selection ($C=72\text{h}$) |
| **`REALISTIC_ACHIEVABLE_UTILITY`** | **`-0.198307`** | `[-0.218529, -0.178330]` | YES | YES | **Deployable Policy** | Deployable utility under locked predictability model ($AUPRC=0.2653$) |
| **`HINDSIGHT_GRID_SCORE_POLICY_CEILING`** | **`-0.198307`** | `[-0.218529, -0.178330]` | YES | YES | Hindsight Policy Sweep | Global peak utility across 2D threshold $\times$ cooldown sweep ($C=72\text{h}$) |
| **`FROZEN_MODEL_UTILITY`** | **`-0.257312`** | `[-0.282823, -0.233519]` | YES | YES | **Primary Deployable Policy** | Fixed deployable policy at prespecified threshold ($th=0.190, C=36\text{h}$) |
| **`ORACLE_TO_GLOBAL_POLICY_UTILITY_GAP`** | **`+1.024585`** ($p < 0.0001$) | `[+0.999690, +1.049449]` | YES | YES | Composite Diagnostic Gap | Observed gap between oracle bound and global score policy ceiling |
