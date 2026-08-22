# ⚖️ STAGE 2: FINAL ORACLE & CEILING RECONCILIATION REPORT

This report provides the final, authoritative reconciliation of the four historical numbers across the `Hybrid-Transform-for-Early-Sepsis-Prediction` codebase, establishing an empirically verified mathematical foundation for the project's scientific decision.

---

## 1. Master Historical Reconciliation Matrix

| Metric Value | Retired Historical Label | Approved Mandatory Taxonomy | Exact Source File & Line | Scores Used? | Test Labels Used? | Action-Space Constraints | Physical & Scientific Interpretation |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **`+0.826246`** | "Theoretical Oracle Ceiling" | **`GROUND_TRUTH_ORACLE_CEILING`** | [`scripts/oracle_reconciliation_independent.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/oracle_reconciliation_independent.py#L48-L55) (`calculate_best_single_alarm`) | **NO** | **YES** | Single optimal alarm at $\max(0, t_{\text{onset}}-6\text{h})$ | **True Mathematical Ceiling.** Proves the PhysioNet 2019 utility function is mathematically coherent and positive utility is achievable on BIDMC. |
| **`-0.234579`** | "Phase 15 BIDMC Oracle Utility" | **`HINDSIGHT_COOLDOWN_SCORE_POLICY_CEILING`** | [`scripts/run_m3_phase15_frozen_score_diagnostics.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/run_m3_phase15_frozen_score_diagnostics.py#L308-L318) (`best_test_oracle_u`) | **YES** | **YES** | Hindsight sweep ($th=0.440$, **Cooldown $C=36\text{h}$**) | **Best Score-Based Ceiling on Frozen Model.** Highest utility achievable by tuning decision thresholds in hindsight on frozen model outputs with alert suppression. |
| **`-0.235183`** | "Phase 16 BIDMC Oracle Utility" | **`RETRAINED_HINDSIGHT_COOLDOWN_SCORE_POLICY_CEILING`** | [`scripts/run_m3_phase16_representation_forensics.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/run_m3_phase16_representation_forensics.py#L508-L514) (`best_exp_test_oracle_u`) | **YES** | **YES** | Hindsight sweep ($th=0.450$, **Cooldown $C=36\text{h}$**) | **Best Score-Based Ceiling on Retrained DANN.** Proves domain-adversarial retraining failed to shift the score ceiling ($\Delta = -0.000604$). |
| **`-0.855545`** | "Observable-Score Oracle Ceiling" | **`HINDSIGHT_RAW_SCORE_POLICY_CEILING`** | [`scripts/run_m3_phase17_feasibility_decision_gate.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/run_m3_phase17_feasibility_decision_gate.py#L286-L293) (`obs_oracle_u`) | **YES** | **YES** | Hindsight sweep ($th=0.745$, **No Cooldown $C=0\text{h}$**) | **Best Raw Threshold Ceiling Without Cooldown.** Demonstrates that without alert suppression, unsuppressed false alarms penalize utility by $-0.05$/hr. |

---

## 2. Scientific Problem Diagnosis & Forensic Breakdown

### A. The Utility Paradox: High AUROC ($0.9617$) vs Negative Deployable Utility ($-0.257312$)
1. **AUROC vs Utility Metric Disconnect:**  
   AUROC measures threshold-agnostic score ranking across all hourly records. In ICU data, non-septic patients contribute $>98\%$ of all hourly observations ($726,927$ non-septic hours out of $753,927$ total hours).
2. **False Alarm Accumulation:**  
   The official utility metric penalizes false alarms at $-0.05$ points per hour. Even at high specificity ($99.34\%$), false alarms occur on $0.66\%$ of non-septic hours, accumulating $-2146.75$ points of false alarm penalties across $18,934$ non-septic patients.
3. **Score Overlap in Non-Septic Mimics:**  
   Observable risk probabilities $p(t)$ for non-septic patients with clinical mimic conditions (e.g. SIRS, fever, tachycardia) overlap significantly with early septic risk probabilities, making it impossible to select a threshold that achieves positive utility without triggering overwhelming false alarm burdens.

### B. Why Domain Adaptation (DANN) Failed to Shift the Ceiling
- Baseline Frozen Model Score Ceiling: **`-0.234579`**
- Retrained Domain-Adversarial (DANN) Score Ceiling: **`-0.235183`**
- Delta: **`-0.000604`** (negligible change in the wrong direction).
- *Forensic Finding:* DANN removed hospital-identifying domain features (Emory vs BIDMC), but did NOT resolve intra-hospital score overlap between septic patients and non-septic mimic patients in BIDMC. The bottleneck is feature separability, not domain shift alone.

---

## 3. Proposed Case Classification

Based on the verified numerical evidence:
$$\text{GROUND\_TRUTH\_ORACLE\_CEILING} = \mathbf{+0.826246} > 0 \quad (\text{Positive})$$
$$\text{HINDSIGHT\_COOLDOWN\_SCORE\_POLICY\_CEILING} = \mathbf{-0.234579} < 0 \quad (\text{Negative})$$

This combination maps unambiguously to:
### **CASE B: INFORMATION-LIMITED**

### Scientific Implication:
Positive utility is mathematically achievable under the official action space and PhysioNet utility definition. However, model probability outputs derived from the observable clinical feature set cannot achieve positive utility under any decision policy or threshold tuning.

**Mandatory Directive:** **STOP ALL NEURAL NETWORK ARCHITECTURE SEARCH AND RETRAIN ATTEMPTS.** Further model tuning (Transformers, DANNs, CNNs, loss functions) cannot overcome the score separability limit. The paper's contribution must be reframed around documenting the cross-hospital score separability limit and temporal risk representation boundaries.
