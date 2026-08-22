# 📋 MANUSCRIPT NUMERICAL AUDIT & CLAIMS CHECKLIST

This document provides a strict 1-to-1 numerical audit verifying every metric reported in the manuscript against authoritative Phase 18 artifacts, followed by a non-negotiable checklist of prohibited claims.

---

## 1. Manuscript Numerical Audit Table

| Manuscript Value | Metric Taxonomy | Primary Source Artifact | Verification Status | Dataset / Split | Deployable or Hindsight? | Paper Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **`+0.826246`** (`+0.826245570148`) | `GROUND_TRUTH_ORACLE_CEILING` | [`reports/oracle_reconciliation/source_inventory.md`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/reports/oracle_reconciliation/source_inventory.md) | **VERIFIED EXACT** ($\le 10^{-10}$) | BIDMC Test ($N=20,000$) | Hindsight Theoretical | Primary Benchmark Ceiling |
| **`-0.257312`** (`-0.257312450379`) | `FROZEN_MODEL_UTILITY` | [`results/m3_final_test_predictions.npz`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/results/m3_final_test_predictions.npz) | **VERIFIED EXACT** ($\le 10^{-10}$) | BIDMC Test ($N=20,000$) | **Deployable** ($th=0.190, C=36\text{h}$) | Primary Deployable Result |
| **`-0.198307`** | `HINDSIGHT_GRID_SCORE_POLICY_CEILING` | [`results/oracle_reconciliation/extended_cooldown_grid.csv`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/results/oracle_reconciliation/extended_cooldown_grid.csv) | **VERIFIED EXACT** | BIDMC Test ($N=20,000$) | Hindsight ($th=0.345, C=72\text{h}$) | Diagnostic Policy Ceiling |
| **`+0.281895`** | `PATIENT_ADAPTIVE_THRESHOLD_CEILING` | [`results/oracle_reconciliation/patient_adaptive_ceiling_v2.csv`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/results/oracle_reconciliation/patient_adaptive_ceiling_v2.csv) | **VERIFIED EXACT** | BIDMC Test ($N=20,000$) | Counterfactual Hindsight ($C=72\text{h}$) | Diagnostic Headroom Bound |
| **`-0.198307`** | `REALISTIC_ACHIEVABLE_UTILITY` | [`reports/oracle_reconciliation/adaptive_threshold_predictability.md`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/reports/oracle_reconciliation/adaptive_threshold_predictability.md) | **VERIFIED EXACT** | BIDMC Test ($N=20,000$) | Deployable ($AUPRC=0.2653$) | Realistic Achievable Bound |
| **`-0.855545`** | `RAW_SCORE_POLICY_CEILING` | [`reports/oracle_reconciliation/source_inventory.md`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/reports/oracle_reconciliation/source_inventory.md) | **VERIFIED EXACT** | BIDMC Test ($N=20,000$) | Hindsight ($th=0.745, C=0\text{h}$) | Action-Space Diagnostic |
| **`0.961726`** (`0.9617`) | Test AUROC | [`reports/final_decision/reproducibility_manifest.md`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/reports/final_decision/reproducibility_manifest.md) | **VERIFIED EXACT** | BIDMC Test ($N=20,000$) | Deployable | Primary Discrimination Metric |
| **`0.423114`** (`0.4231`) | Test AUPRC | [`reports/final_decision/reproducibility_manifest.md`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/reports/final_decision/reproducibility_manifest.md) | **VERIFIED EXACT** | BIDMC Test ($N=20,000$) | Deployable | Primary Discrimination Metric |
| **`0.015290`** | Test Brier Score | [`reports/final_decision/reproducibility_manifest.md`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/reports/final_decision/reproducibility_manifest.md) | **VERIFIED EXACT** | BIDMC Test ($N=20,000$) | Deployable | Calibration Metric |
| **`0.018151`** | Test ECE | [`reports/final_decision/reproducibility_manifest.md`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/reports/final_decision/reproducibility_manifest.md) | **VERIFIED EXACT** | BIDMC Test ($N=20,000$) | Deployable | Calibration Metric |
| **`[+0.806653, +0.844781]`** | GT Oracle 95% Bootstrap CI | [`results/oracle_reconciliation/bootstrap_ci_all_metrics.csv`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/results/oracle_reconciliation/bootstrap_ci_all_metrics.csv) | **VERIFIED EXACT** | BIDMC Test ($B=1,000$) | Hindsight | Uncertainty Quantification |
| **`[-0.282823, -0.233519]`** | Frozen Model 95% Bootstrap CI | [`results/oracle_reconciliation/bootstrap_ci_all_metrics.csv`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/results/oracle_reconciliation/bootstrap_ci_all_metrics.csv) | **VERIFIED EXACT** | BIDMC Test ($B=1,000$) | Deployable | Uncertainty Quantification |
| **`[-0.218529, -0.178330]`** | Grid Policy 95% Bootstrap CI | [`results/oracle_reconciliation/bootstrap_ci_all_metrics.csv`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/results/oracle_reconciliation/bootstrap_ci_all_metrics.csv) | **VERIFIED EXACT** | BIDMC Test ($B=1,000$) | Hindsight | Uncertainty Quantification |
| **`+0.538943`** ($p < 0.0001$) | Paired $\Delta$ (Adaptive - Frozen) | [`results/oracle_reconciliation/paired_significance_tests.csv`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/results/oracle_reconciliation/paired_significance_tests.csv) | **VERIFIED EXACT** | BIDMC Test ($B=1,000$) | Paired Comparison | Significance Test |
| **`+1.024585`** ($p < 0.0001$) | Paired $\Delta$ (GT - Grid Ceiling) | [`results/oracle_reconciliation/paired_significance_tests.csv`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/results/oracle_reconciliation/paired_significance_tests.csv) | **VERIFIED EXACT** | BIDMC Test ($B=1,000$) | Paired Comparison | Significance Test |
| **`0.9609 ± 0.0016`** | 6-Seed AUROC Mean $\pm$ Std | [`results/multiseed/multiseed_summary.csv`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/results/multiseed/multiseed_summary.csv) | **VERIFIED EXACT** | BIDMC Test (6 Seeds) | Multi-seed | Stability Check |
| **`-0.257316 ± 0.002012`** | 6-Seed Utility Mean $\pm$ Std | [`results/multiseed/multiseed_summary.csv`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/results/multiseed/multiseed_summary.csv) | **VERIFIED EXACT** | BIDMC Test (6 Seeds) | Multi-seed | Stability Check |

---

## 2. "Claims That Must NOT Be Made" Checklist

- [x] **DO NOT claim 100% statistical confidence.** Use *"the 95% patient-level bootstrap confidence interval remained entirely below zero"*.
- [x] **DO NOT call test threshold sweeps an "oracle".** The term *"Post-Hoc Test Threshold Sweep Oracle"* is retired; use `HINDSIGHT_GRID_SCORE_POLICY_CEILING`.
- [x] **DO NOT claim universal cross-hospital generalization.** The study evaluates two major US academic medical centers (Emory and BIDMC).
- [x] **DO NOT claim that Transformers are fundamentally incapable of clinical utility.** State that for the evaluated $M3$ representation and BIDMC setting, high discrimination did not yield positive utility.
- [x] **DO NOT present `PATIENT_ADAPTIVE_THRESHOLD_CEILING` (+0.281895) as deployable performance.** It is explicitly labeled a counterfactual diagnostic ceiling.
- [x] **DO NOT report $p = 0.0000$ as zero.** Report $p < 0.0001$ ($p < 1/B$).
- [x] **DO NOT claim the decomposition framework is "the first-ever".** Use *"we propose"* or *"we introduce"*.
