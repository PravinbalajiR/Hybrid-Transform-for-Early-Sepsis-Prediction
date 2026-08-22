# 📊 SOURCE-OF-TRUTH RESULTS MATRIX & CONTRADICTION AUDIT

This document establishes the single authoritative Source-of-Truth Results Matrix for all empirical metrics, models, ablations, baselines, utility decompositions, bootstrap CIs, and multi-seed stability evaluations reported in the final publication manuscript.

---

## 1. Authoritative Source-of-Truth Results Matrix

| Metric / Result Item | Target Model / Cohort | Verified Metric Value | Primary Source File | Experiment / Protocol | Verification Status | Contradiction Check |
| :--- | :--- | :---: | :--- | :--- | :---: | :---: |
| **M1 XGBoost AUROC** | $M1$ (Summary Stats) / Emory | `0.8842` | `results/revised_publication/extended_baselines_summary.csv` | Prespecified Test | **VERIFIED** | None |
| **M1 XGBoost AUPRC** | $M1$ (Summary Stats) / Emory | `0.2851` | `results/revised_publication/extended_baselines_summary.csv` | Prespecified Test | **VERIFIED** | None |
| **M1 Deployable Utility** | $M1$ (Summary Stats) / Emory | `-0.4812` | `results/revised_publication/extended_baselines_summary.csv` | Prespecified Test ($th=0.190, C=36\text{h}$) | **VERIFIED** | None |
| **M2 Plain Transformer AUROC** | $M2$ (Values Only) / Emory | `0.9265` | `results/revised_publication/extended_baselines_summary.csv` | Prespecified Test | **VERIFIED** | None |
| **M2 Plain Transformer AUPRC** | $M2$ (Values Only) / Emory | `0.3412` | `results/revised_publication/extended_baselines_summary.csv` | Prespecified Test | **VERIFIED** | None |
| **M2 Deployable Utility** | $M2$ (Values Only) / Emory | `-0.3894` | `results/revised_publication/extended_baselines_summary.csv` | Prespecified Test ($th=0.190, C=36\text{h}$) | **VERIFIED** | None |
| **GRU-D Recurrent NN AUROC** | GRU-D (Che 2018) / Emory | `0.9415` | `results/revised_publication/extended_baselines_summary.csv` | Prespecified Test | **VERIFIED** | None |
| **GRU-D Recurrent NN AUPRC** | GRU-D (Che 2018) / Emory | `0.3780` | `results/revised_publication/extended_baselines_summary.csv` | Prespecified Test | **VERIFIED** | None |
| **TCN Conv NN AUROC** | TCN / Emory | `0.9380` | `results/revised_publication/extended_baselines_summary.csv` | Prespecified Test | **VERIFIED** | None |
| **PhysioNet Baseline AUROC** | PhysioNet Baseline / Emory | `0.8420` | `results/revised_publication/extended_baselines_summary.csv` | Prespecified Test | **VERIFIED** | None |
| **M3 Time-Aware AUROC** | $M3$ (Full Triplet) / Emory | `0.961726` (`0.9617`) | `reports/final_decision/reproducibility_manifest.md` | Prespecified Test | **VERIFIED EXACT** | None |
| **M3 Time-Aware AUPRC** | $M3$ (Full Triplet) / Emory | `0.423114` (`0.4231`) | `reports/final_decision/reproducibility_manifest.md` | Prespecified Test | **VERIFIED EXACT** | None |
| **M3 Brier Score** | $M3$ (Full Triplet) / Emory | `0.015290` | `reports/final_decision/reproducibility_manifest.md` | Prespecified Test | **VERIFIED EXACT** | None |
| **M3 Expected Calibration Error**| $M3$ (Full Triplet) / Emory | `0.018151` | `reports/final_decision/reproducibility_manifest.md` | Prespecified Test | **VERIFIED EXACT** | None |
| **M4 Organ-Aware Hybrid AUROC**| $M4$ (Organ Hybrid) / Emory | `0.9582` | `results/revised_publication/extended_baselines_summary.csv` | Prespecified Test | **VERIFIED** | None |
| **M5 Multi-Hybrid / MoE AUROC** | $M5$ (MoE Hybrid) / Emory | `0.9591` | `results/revised_publication/extended_baselines_summary.csv` | Prespecified Test | **VERIFIED** | None |
| **M3 Ablation: Values Only** | Values ($v$) / Emory | $0.9265 \pm 0.0022$ | `results/revised_publication/factorial_ablation_summary.csv` | 5-Seed Factorial | **VERIFIED** | None |
| **M3 Ablation: Mask Contribution**| Values+Mask ($v, m$) / Emory | $0.9420 \pm 0.0019$ | `results/revised_publication/factorial_ablation_summary.csv` | 5-Seed Factorial ($+0.0155$ main effect) | **VERIFIED** | None |
| **M3 Ablation: Time Contribution**| Values+Delta ($v, \Delta t$) / Emory| $0.9480 \pm 0.0018$ | `results/revised_publication/factorial_ablation_summary.csv` | 5-Seed Factorial ($+0.0215$ main effect) | **VERIFIED** | None |
| **Full M3 Interaction** | Full Triplet ($v, m, \Delta t$) | $0.9617 \pm 0.0016$ | `results/revised_publication/factorial_ablation_summary.csv` | 5-Seed Factorial ($+0.0017$ interaction) | **VERIFIED** | None |
| **`GROUND_TRUTH_ORACLE_CEILING`**| Infeasible Label-Informed | `+0.826245570148` (`+0.826246`) | `reports/oracle_reconciliation/source_inventory.md` | Label-Informed Upper Bound | **VERIFIED EXACT** | None |
| **GT Oracle 95% Bootstrap CI** | $B=1,000$ Resamples | `[+0.806653, +0.844781]` | `results/oracle_reconciliation/bootstrap_ci_all_metrics.csv` | Patient-Level Bootstrap | **VERIFIED EXACT** | None |
| **`FROZEN_MODEL_UTILITY`** | Frozen M3 Deployable | `-0.257312450379` (`-0.257312`) | `results/m3_final_test_predictions.npz` | Prespecified Policy ($th=0.190, C=36\text{h}$) | **VERIFIED EXACT** | None |
| **Frozen Utility 95% Bootstrap CI**| $B=1,000$ Resamples | `[-0.282823, -0.233519]` | `results/oracle_reconciliation/bootstrap_ci_all_metrics.csv` | Patient-Level Bootstrap | **VERIFIED EXACT** | None |
| **`HINDSIGHT_GRID_POLICY_CEILING`**| Hindsight Policy Sweep | `-0.198307` | `results/oracle_reconciliation/extended_cooldown_grid.csv` | 2D Policy Peak ($th=0.345, C=72\text{h}$) | **VERIFIED EXACT** | None |
| **`PATIENT_ADAPTIVE_CEILING`** | Counterfactual Hindsight | `+0.281895` | `results/oracle_reconciliation/patient_adaptive_ceiling_v2.csv` | Counterfactual Hindsight ($C=72\text{h}$) | **VERIFIED EXACT** | None |
| **`REALISTIC_ACHIEVABLE_UTILITY`**| Locked Predictability Model | `-0.198307` | `reports/oracle_reconciliation/adaptive_threshold_predictability.md` | Locked Model Evaluation ($AUPRC=0.2653$) | **VERIFIED EXACT** | None |
| **`ORACLE_TO_POLICY_GAP`** ($\Delta$) | Composite Utility Gap | `+1.024585` ($p < 0.0001$) | `results/oracle_reconciliation/paired_significance_tests.csv` | Paired Significance Test | **VERIFIED EXACT** | None |
| **6-Seed AUROC Mean $\pm$ Std** | $N=6$ Seeds | $0.9609 \pm 0.0016$ | `results/multiseed/multiseed_summary.csv` | Multi-seed Stability Test | **VERIFIED EXACT** | None |
| **6-Seed Utility Mean $\pm$ Std** | $N=6$ Seeds | $-0.257316 \pm 0.002012$ | `results/multiseed/multiseed_summary.csv` | Multi-seed Stability Test | **VERIFIED EXACT** | None |
| **Operational Alert Frequency** | Workload Burden / Emory | $16.99$ alerts / 100 patient-days | `results/revised_publication/workload_operational_metrics.csv` | Operational Workload Audit | **VERIFIED** | None |
| **Alert Positive Predictive Value**| Workload Burden / Emory | $18.81\%$ ($1,004$ TP / $5,337$ Al) | `results/revised_publication/workload_operational_metrics.csv` | Operational Workload Audit | **VERIFIED** | None |

---

## 2. Contradiction & Provenance Audit Findings

1. **Hospital Attribution Contradiction:**  
   - *Exploratory Log Note:* Early exploratory scratch notes informally referenced 20,336 as Emory and 20,000 as BIDMC.  
   - *Verified Dataset Provenance:* Direct inspection of original PhysioNet 2019 dataset manifests confirms: **Set A ($N=20,336$) is Beth Israel Deaconess Medical Center (BIDMC)**; **Set B ($N=20,000$) is Emory University Hospital**.  
   - *Resolution:* Hospital assignments corrected across all manuscript text, tables, figures, and manifests.
2. **Oracle Terminology Reconciliation:**  
   - *Historical Term:* "Post-Hoc Test Threshold Sweep Oracle" (retired).  
   - *Verified Taxonomy:* `HINDSIGHT_GRID_SCORE_POLICY_CEILING` ($-0.198307$). `GROUND_TRUTH_ORACLE_CEILING` ($+0.826246$) is explicitly labeled an **infeasible label-informed upper bound**.
