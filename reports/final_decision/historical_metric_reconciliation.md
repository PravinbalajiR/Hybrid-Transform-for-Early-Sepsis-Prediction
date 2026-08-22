# 📜 HISTORICAL METRIC RECONCILIATION REPORT (TASK 6)

This document provides a comprehensive, side-by-side reconciliation of all seven historical numerical metrics reported across Phases 14–17 of the `Hybrid-Transform-for-Early-Sepsis-Prediction` project.

---

## 📊 Master Historical Metric Reconciliation Matrix

| Metric Value | Exact Approved Taxonomy | Primary Source Script | Model Dependent? | Label Dependent? | Action Space & Policy Constraints | Hindsight Optimized? | Deployable Clinical Policy? | True Oracle? | Main Paper Status |
| :---: | :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| **`+0.826246`** | **`GROUND_TRUTH_ORACLE_CEILING`** | [`scripts/oracle_reconciliation_independent.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/oracle_reconciliation_independent.py#L48-L55) | **NO** | **YES** | Single optimal alarm at $\max(0, t_{\text{onset}}-6\text{h})$ | **NO** | **NO** | **YES** | Primary Benchmark Ceiling |
| **`-0.257312`** | **`FROZEN_MODEL_UTILITY`** | [`scripts/reproduce_final_m3.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/reproduce_final_m3.py#L20-L33) | **YES** | **YES** | Fixed deployable policy ($th=0.190$, **Cooldown $C=36\text{h}$**) | **NO** | **YES** | **NO** | Primary Deployable Result |
| **`-0.234579`** | **`HINDSIGHT_COOLDOWN_SCORE_POLICY_CEILING`** | [`scripts/run_m3_phase15_frozen_score_diagnostics.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/run_m3_phase15_frozen_score_diagnostics.py#L308-L318) | **YES** | **YES** | Hindsight sweep ($th=0.440$, **Cooldown $C=36\text{h}$**) | **YES** | **NO** | **NO** | Diagnostic Upper Bound |
| **`-0.235183`** | **`RETRAINED_HINDSIGHT_COOLDOWN_SCORE_POLICY_CEILING`** | [`scripts/run_m3_phase16_representation_forensics.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/run_m3_phase16_representation_forensics.py#L508-L514) | **YES** | **YES** | Retrained DANN sweep ($th=0.450$, **Cooldown $C=36\text{h}$**) | **YES** | **NO** | **NO** | Retrained DANN Bound |
| **`-0.855545`** | **`RAW_SCORE_POLICY_CEILING`** | [`scripts/run_m3_phase17_feasibility_decision_gate.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/run_m3_phase17_feasibility_decision_gate.py#L286-L293) | **YES** | **YES** | Hindsight sweep ($th=0.745$, **No Cooldown $C=0\text{h}$**) | **YES** | **NO** | **NO** | Action Space Diagnostic |
| **`-0.198307`** | **`HINDSIGHT_GRID_SCORE_POLICY_CEILING`** | [`scripts/run_oracle_reconciliation_extended.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/run_oracle_reconciliation_extended.py#L55-L95) | **YES** | **YES** | 2D policy sweep ($th=0.345$, **Cooldown $C=72\text{h}$**) | **YES** | **NO** | **NO** | 2D Policy Peak Ceiling |
| **`+0.281895`** | **`PATIENT_ADAPTIVE_THRESHOLD_CEILING`** | [`scripts/run_patient_adaptive_ceiling_v2.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/run_patient_adaptive_ceiling_v2.py#L40-L75) | **YES** | **YES** | Per-patient optimal threshold ($C=72\text{h}$) | **YES** | **NO** | **NO** | Counterfactual Ceiling |

---

## 🔍 Comprehensive Numerical Reconciliation Notes

1. **`+0.826246` (`GROUND_TRUTH_ORACLE_CEILING`):**  
   Exact ratio: $\frac{880.777778}{1066.0} = \mathbf{+0.826245570148}$. Derived purely from $y_{\text{true}}$ and $t_{\text{onset}}$. Zero model predictions, logits, or thresholds involved. Proves the PhysioNet 2019 utility metric is mathematically coherent.

2. **`-0.257312` (`FROZEN_MODEL_UTILITY`):**  
   Primary deployable result of frozen M3 Transformer on held-out BIDMC test set ($N=20,000$) evaluated at prespecified validation threshold $th=0.190$ and $C=36\text{h}$ cooldown.

3. **`-0.234579` (`HINDSIGHT_COOLDOWN_SCORE_POLICY_CEILING`):**  
   Peak post-hoc threshold sweep utility on frozen M3 test probabilities under 36h cooldown alert suppression ($th=0.440$).

4. **`-0.235183` (`RETRAINED_HINDSIGHT_COOLDOWN_SCORE_POLICY_CEILING`):**  
   Peak post-hoc threshold sweep utility on retrained Domain-Adversarial Neural Network (DANN Exp F) probabilities under 36h cooldown ($th=0.450$). Differs by $\Delta = -0.000604$ due to retrained neural weights.

5. **`-0.855545` (`RAW_SCORE_POLICY_CEILING`):**  
   Peak raw threshold sweep utility without alert suppression ($C=0\text{h}, th=0.745$). Demonstrates false alarm accumulation penalties ($-0.05$ pts/hr) on unsuppressed predictions.

6. **`-0.198307` (`HINDSIGHT_GRID_SCORE_POLICY_CEILING`):**  
   Global peak utility across extended 2D grid search ($C \in \{6, 12, 24, 36, 48, 72, 96, 120, 144, 168, 240, 336, C_{\text{MAX}}\}\text{h} \times th \in [0.005, 0.995]$). Utility turns over at $C=72\text{h}$ ($th=0.345$).

7. **`+0.281895` (`PATIENT_ADAPTIVE_THRESHOLD_CEILING`):**  
   Counterfactual non-deployable diagnostic ceiling where each patient's threshold $th_i^*$ is chosen in hindsight using full trajectory and outcome knowledge.
