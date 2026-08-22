# 🔒 SUPPLEMENTARY MATERIAL & REPRODUCIBILITY MANIFEST

---

## APPENDIX A: HISTORICAL METRIC RECONCILIATION TABLE

This document provides a side-by-side reconciliation of all historical numerical metrics reported across early exploratory phases of the project.

| Historical Value | Historical Label | Approved Mandatory Taxonomy | Hindsight Status | Primary Source Artifact | Explanation of Discrepancy / Action Space |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **`+0.826246`** | "Theoretical Oracle Ceiling" | **`GROUND_TRUTH_ORACLE_CEILING`** | **NO** | `source_inventory.md` | Ground-truth only. $880.78 / 1066.0$ pts. Infeasible. |
| **`-0.257312`** | "BIDMC Deployable Utility" | **`FROZEN_MODEL_UTILITY`** | **NO** | `m3_final_test_predictions.npz` | Fixed deployable policy ($th=0.190, C=36\text{h}$) on Emory test set. |
| **`-0.234579`** | "Phase 15 Test Oracle" | **`HINDSIGHT_GRID_SCORE_POLICY_CEILING`** | **YES** | `run_m3_phase15_frozen_score_diagnostics.py` | Hindsight sweep ($th=0.440, C=36\text{h}$). Retired "oracle" label. |
| **`-0.235183`** | "Phase 16 Retrained DANN" | **`RETRAINED_HINDSIGHT_COOLDOWN_CEILING`** | **YES** | `run_m3_phase16_representation_forensics.py` | Retrained DANN sweep ($th=0.450, C=36\text{h}$). $\Delta = -0.000604$. |
| **`-0.855545`** | "Observable Score Ceiling" | **`RAW_SCORE_POLICY_CEILING`** | **YES** | `run_m3_phase17_feasibility_decision_gate.py` | Hindsight sweep without alert suppression ($C=0\text{h}, th=0.745$). |
| **`-0.198307`** | "Extended Grid Peak" | **`HINDSIGHT_GRID_SCORE_POLICY_CEILING`** | **YES** | `extended_cooldown_grid.csv` | 2D policy sweep global peak ($th=0.345, C=72\text{h}$). |
| **`+0.281895`** | "Patient-Adaptive Ceiling" | **`PATIENT_ADAPTIVE_THRESHOLD_CEILING`** | **YES** | `patient_adaptive_ceiling_v2.csv` | Counterfactual per-patient threshold selection ($C=72\text{h}$). Infeasible. |

---

## APPENDIX B: EXTENDED 2D POLICY SURFACES & WORKLOAD DETAIL

Net utility $U(th, C)$ across alert suppression cooldowns $C \in \{6, 12, 24, 36, 48, 72, 96, 120, 144, 168, 240, 336\}\text{h}$:
- $C = 6\text{h}$: Peak $U = -0.669864$ (at $th = 0.650$)
- $C = 12\text{h}$: Peak $U = -0.499808$ (at $th = 0.580$)
- $C = 24\text{h}$: Peak $U = -0.320492$ (at $th = 0.520$)
- $C = 36\text{h}$: Peak $U = -0.234579$ (at $th = 0.440$)
- $C = 48\text{h}$: Peak $U = -0.201646$ (at $th = 0.380$)
- **$C = 72\text{h}$:** **Global Peak $U = -0.198307$ (at $th = 0.345$) [INTERIOR MAXIMUM]**
- $C = 96\text{h}$: Peak $U = -0.199658$ (Decreased due to true positive suppression)
- $C = 168\text{h}$: Peak $U = -0.202908$ (Decreased)

---

## APPENDIX C: CRYPTOGRAPHIC SHA256 MANIFEST

| Artifact / File Name | File Path | SHA256 Cryptographic Hash |
| :--- | :--- | :--- |
| **Frozen M3 Checkpoint** | `experiments/final_m3_frozen/best_m3_frozen.pt` | `5b22607444f4a242a52d0d9337e60c4c63044542dc6796a4a9de78c5ef38057c` |
| **Emory Test Predictions NPZ** | `results/m3_final_test_predictions.npz` | `e4a6a5e171b3e94bd2d6b38c2ef40eb14032d91c1b3f9ffc129e9ae70678ed70` |
| **BIDMC Val Predictions NPZ** | `results/m3_final_val_predictions.npz` | `c3ee258be16e11894d38e219fb099fef4c0dceaeedec0b37493a778b4a7ee5f7` |
| **Test Split Manifest** | `data/splits/test_ids.json` | `55d5bc58000bc19e59d9eef27ca5f5d81bdab7ed74a88f7b764c0173adbd923b` |
| **Independent Utility Calculator** | `scripts/oracle_reconciliation_independent.py` | `0f0bf6085a815a5fbc4001923cb82dc99a2fbde6097561f7481a53a9cd388a10` |
