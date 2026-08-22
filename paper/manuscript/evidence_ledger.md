# 📓 INTERNAL EVIDENCE LEDGER

This Evidence Ledger maps every numerical claim, architectural result, and operational metric in the unified paper to its authoritative Phase 18 verified source artifact.

---

## 1. Authoritative Evidence Ledger Table

| Claim / Metric | Value / Estimate | Source Artifact | Dataset Split | Test/Val | Prespecified / Post-hoc | Verification Status |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **M1 (XGBoost) AUROC** | `0.8842` | `results/phase16/phase16_architecture_differences.csv` | Set B (BIDMC) | Test | Prespecified | Verified |
| **M1 (XGBoost) AUPRC** | `0.2851` | `results/phase16/phase16_architecture_differences.csv` | Set B (BIDMC) | Test | Prespecified | Verified |
| **M2 (Plain Transformer) AUROC** | `0.9265` | `results/phase16/phase16_architecture_differences.csv` | Set B (BIDMC) | Test | Prespecified | Verified |
| **M2 (Plain Transformer) AUPRC** | `0.3412` | `results/phase16/phase16_architecture_differences.csv` | Set B (BIDMC) | Test | Prespecified | Verified |
| **M3 (Time-Aware) AUROC** | `0.961726` (`0.9617`) | `reports/final_decision/reproducibility_manifest.md` | Set B (BIDMC) | Test | Prespecified | Verified Exact |
| **M3 (Time-Aware) AUPRC** | `0.423114` (`0.4231`) | `reports/final_decision/reproducibility_manifest.md` | Set B (BIDMC) | Test | Prespecified | Verified Exact |
| **M3 Brier Score** | `0.015290` | `reports/final_decision/reproducibility_manifest.md` | Set B (BIDMC) | Test | Prespecified | Verified Exact |
| **M3 ECE** | `0.018151` | `reports/final_decision/reproducibility_manifest.md` | Set B (BIDMC) | Test | Prespecified | Verified Exact |
| **M3 Ablation: No-Time + No-Mask** | `0.9265` (AUROC) | `results/phase16/phase16_ablation.csv` | Set B (BIDMC) | Test | Controlled | Verified |
| **M3 Ablation: Time-Aware Only** | `0.9480` (AUROC) | `results/phase16/phase16_ablation.csv` | Set B (BIDMC) | Test | Controlled | Verified |
| **M3 Ablation: Mask-Aware Only** | `0.9420` (AUROC) | `results/phase16/phase16_ablation.csv` | Set B (BIDMC) | Test | Controlled | Verified |
| **M3 Full (Time + Mask)** | `0.9617` (AUROC) | `results/phase16/phase16_ablation.csv` | Set B (BIDMC) | Test | Controlled | Verified |
| **M4 (Organ-Aware Hybrid) AUROC** | `0.9582` | `results/phase16/phase16_architecture_differences.csv` | Set B (BIDMC) | Test | Prespecified | Verified |
| **M5 (Multi-Hybrid / MoE) AUROC** | `0.9591` | `results/phase16/phase16_architecture_differences.csv` | Set B (BIDMC) | Test | Prespecified | Verified |
| **`GROUND_TRUTH_ORACLE_CEILING`** | `+0.826245570148` (`+0.826246`) | `reports/oracle_reconciliation/source_inventory.md` | Set B (BIDMC) | Test | Hindsight Ground-Truth | Verified Exact ($\le 10^{-10}$) |
| **GT Oracle 95% Bootstrap CI** | `[+0.806653, +0.844781]` | `results/oracle_reconciliation/bootstrap_ci_all_metrics.csv` | Set B ($B=1000$) | Test | Patient-level Bootstrap | Verified Exact |
| **`FROZEN_MODEL_UTILITY`** | `-0.257312450379` (`-0.257312`) | `results/m3_final_test_predictions.npz` | Set B (BIDMC) | Test | Prespecified ($th=0.190, C=36\text{h}$) | Verified Exact ($\le 10^{-10}$) |
| **Frozen Utility 95% Bootstrap CI** | `[-0.282823, -0.233519]` | `results/oracle_reconciliation/bootstrap_ci_all_metrics.csv` | Set B ($B=1000$) | Test | Patient-level Bootstrap | Verified Exact |
| **`HINDSIGHT_GRID_SCORE_POLICY_CEILING`** | `-0.198307` | `results/oracle_reconciliation/extended_cooldown_grid.csv` | Set B (BIDMC) | Test | Hindsight ($th=0.345, C=72\text{h}$) | Verified Exact |
| **Grid Policy 95% Bootstrap CI** | `[-0.218529, -0.178330]` | `results/oracle_reconciliation/bootstrap_ci_all_metrics.csv` | Set B ($B=1000$) | Test | Patient-level Bootstrap | Verified Exact |
| **`PATIENT_ADAPTIVE_THRESHOLD_CEILING`** | `+0.281895` | `results/oracle_reconciliation/patient_adaptive_ceiling_v2.csv` | Set B (BIDMC) | Test | Counterfactual Hindsight ($C=72\text{h}$) | Verified Exact |
| **Adaptive Ceiling 95% Bootstrap CI** | `[+0.257904, +0.303975]` | `results/oracle_reconciliation/bootstrap_ci_all_metrics.csv` | Set B ($B=1000$) | Test | Patient-level Bootstrap | Verified Exact |
| **`REALISTIC_ACHIEVABLE_UTILITY`** | `-0.198307` | `reports/oracle_reconciliation/adaptive_threshold_predictability.md` | Set B (BIDMC) | Test | Deployable ($AUPRC=0.2653$) | Verified Exact |
| **`RAW_SCORE_POLICY_CEILING`** | `-0.855545` | `reports/oracle_reconciliation/source_inventory.md` | Set B (BIDMC) | Test | Hindsight ($th=0.745, C=0\text{h}$) | Verified Exact |
| **Paired $\Delta_{\text{Adaptive - Frozen}}$** | `+0.538943` ($p < 0.0001$) | `results/oracle_reconciliation/paired_significance_tests.csv` | Set B ($B=1000$) | Test | Significance Test | Verified Exact |
| **Paired $\Delta_{\text{GT - Grid Ceiling}}$** | `+1.024585` ($p < 0.0001$) | `results/oracle_reconciliation/paired_significance_tests.csv` | Set B ($B=1000$) | Test | Significance Test | Verified Exact |
| **6-Seed AUROC Mean $\pm$ Std** | `0.9609 ± 0.0016` | `results/multiseed/multiseed_summary.csv` | Set B (6 Seeds) | Test | Multi-seed Stability | Verified Exact |
| **6-Seed Utility Mean $\pm$ Std** | `-0.257316 ± 0.002012` | `results/multiseed/multiseed_summary.csv` | Set B (6 Seeds) | Test | Multi-seed Stability | Verified Exact |
| **Early Trajectory Predictability AUPRC** | `0.2653` (Base Rate `0.2608`) | `reports/oracle_reconciliation/adaptive_threshold_predictability.md` | Set B (BIDMC) | Test | Early Trajectory ($t \in [0, 5]$) | Verified Exact |
