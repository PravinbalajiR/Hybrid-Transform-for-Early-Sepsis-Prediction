# 📊 AUTHORITATIVE SOURCE-OF-TRUTH RESULTS MATRIX (OFFICIAL PHYSIONET 2019 METRIC)

This document establishes the single authoritative Source-of-Truth Results Matrix for all empirical metrics, models, ablations, baselines, utility decompositions, bootstrap CIs, and multi-seed stability evaluations reported in the final publication manuscript.

---

## 1. Authoritative Source-of-Truth Results Matrix

| Metric / Result Item | Target Model / Cohort | Verified Metric Value | Primary Source File | Experiment / Protocol | Verification Status | Contradiction Check |
| :--- | :--- | :---: | :--- | :--- | :---: | :---: |
| **M3 Time-Aware AUROC** | $M3$ (Full Triplet) / Emory | `0.961726` (`0.9617`) | `m3_final_test_predictions.npz` | Prespecified Test | **VERIFIED EXACT** | None |
| **M3 Time-Aware AUPRC** | $M3$ (Full Triplet) / Emory | `0.423114` (`0.4231`) | `m3_final_test_predictions.npz` | Prespecified Test | **VERIFIED EXACT** | None |
| **M3 Accuracy** | $M3$ (Full Triplet) / Emory | `0.971542` | `evaluate_sepsis_score.py` | Prespecified Test ($th=0.190$) | **VERIFIED EXACT** | None |
| **M3 F-measure** | $M3$ (Full Triplet) / Emory | `0.231804` | `evaluate_sepsis_score.py` | Prespecified Test ($th=0.190$) | **VERIFIED EXACT** | None |
| **M3 Brier Score** | $M3$ (Full Triplet) / Emory | `0.015290` | `reports/final_decision/reproducibility_manifest.md` | Prespecified Test | **VERIFIED EXACT** | None |
| **M3 Expected Calibration Error**| $M3$ (Full Triplet) / Emory | `0.018151` | `reports/final_decision/reproducibility_manifest.md` | Prespecified Test | **VERIFIED EXACT** | None |
| **`GROUND_TRUTH_ORACLE_CEILING`**| Official Normalized Max | `+1.000000` | `evaluate_sepsis_score.py` | Official Normalized Oracle Ceiling | **VERIFIED EXACT** | None |
| **`FROZEN_MODEL_UTILITY`** | Frozen M3 Deployable | `+0.655944` (`+0.6559`) | `evaluate_sepsis_score.py` | Prespecified Policy ($th=0.190$) | **VERIFIED EXACT** | None |
| **`HINDSIGHT_GRID_POLICY_CEILING`**| Hindsight Policy Sweep | `+0.655944` (`+0.6559`) | `official_threshold_sensitivity_sweep.csv` | Threshold Sweep Peak ($th=0.190$) | **VERIFIED EXACT** | None |
| **`PATIENT_ADAPTIVE_CEILING`** | Counterfactual Hindsight | `+0.785000` | `official_threshold_sensitivity_sweep.csv` | Counterfactual Hindsight Upper Bound | **VERIFIED EXACT** | None |
| **6-Seed AUROC Mean $\pm$ Std** | $N=6$ Seeds | $0.9609 \pm 0.0016$ | `results/multiseed/multiseed_summary.csv` | Multi-seed Stability Test | **VERIFIED EXACT** | None |
| **6-Seed Utility Mean $\pm$ Std** | $N=6$ Seeds | $+0.6559 \pm 0.0020$ | `results/multiseed/multiseed_summary.csv` | Multi-seed Stability Test | **VERIFIED EXACT** | None |
| **Operational Alert Frequency** | Workload Burden / Emory | $16.99$ alerts / 100 patient-days | `workload_operational_metrics.csv` | Operational Workload Audit | **VERIFIED** | None |
| **Alert Positive Predictive Value**| Workload Burden / Emory | $18.81\%$ ($1,004$ TP / $5,337$ Al) | `workload_operational_metrics.csv` | Operational Workload Audit | **VERIFIED** | None |
| **Unnormalized Best Utility** | 20,000 Emory Patients | `7298.7778` pts | `evaluate_sepsis_score.py` | Official Evaluator Output | **VERIFIED EXACT** | None |
| **Unnormalized Inaction Utility** | 20,000 Emory Patients | `-9512.4444` pts | `evaluate_sepsis_score.py` | Official Evaluator Output | **VERIFIED EXACT** | None |
| **Normalization Denominator** | 20,000 Emory Patients | `16811.2222` pts | `evaluate_sepsis_score.py` | Official Evaluator Output | **VERIFIED EXACT** | None |

---

## 2. Provenance & Lineage Verification

- **Set A (Development Cohort):** Beth Israel Deaconess Medical Center (BIDMC / Hospital A) ($N=20,336$ ICU stays: $16,192$ train, $4,144$ val).
- **Set B (Cross-Hospital Test Cohort):** Emory University Hospital (Hospital B) ($N=20,000$ ICU stays: $1,066$ septic, $18,934$ non-septic).
- **Transfer Direction:** **BIDMC $\to$ Emory**.
