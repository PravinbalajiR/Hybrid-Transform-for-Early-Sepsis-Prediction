# 🔍 SOURCE INVENTORY & TAXONOMY RECONCILIATION FOR HISTORICAL NUMBERS

This document establishes the exact code origin, calculation mechanism, input dependencies, action-space constraints, and strict taxonomy for the four historical numbers reported across Phases 14–17 of the `Hybrid-Transform-for-Early-Sepsis-Prediction` project.

---

## 🛑 STRICT TAXONOMY & TERMINOLOGY REGULATION

1. **RETIRED TERMINOLOGY:** The phrase *"Post-Hoc Test Threshold Sweep Oracle"* is **PERMANENTLY RETIRED**. Optimizing a decision threshold using held-out test-set outcomes is hindsight parameter tuning, NOT an oracle.
2. **APPROVED TAXONOMY:**
   - **`GROUND_TRUTH_ORACLE_CEILING`**: Uses ONLY true sepsis labels ($y_{\text{true}}$), onset times ($t_{\text{onset}}$), and the official action space. Uses **ZERO** model probabilities, logits, predictions, or learned policy parameters.
   - **`HINDSIGHT_SCORE_POLICY_CEILING`**: Uses model probability predictions ($y_{\text{prob}}$) and tunes decision thresholds or policy parameters in hindsight on held-out test labels.
   - **`FROZEN_MODEL_UTILITY`**: Fixed deployable model utility evaluated at a prespecified validation threshold without test-set tuning.

---

## 1. Number 1: `+0.826246` (`GROUND_TRUTH_ORACLE_CEILING`)

### A. Exact File and Function
- **Full Cohort Direct Computation:** Executed via [`scripts/oracle_reconciliation_independent.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/oracle_reconciliation_independent.py#L48-L55) (`calculate_best_single_alarm`) across all **20,000 BIDMC test patients**.
- **Historical References:** [`scripts/recompute_exact_decompositions.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/recompute_exact_decompositions.py#L180-L208), [`scripts/run_m3_phase14_utr.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/run_m3_phase14_utr.py#L484), [`scripts/run_m3_phase17_feasibility_decision_gate.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/run_m3_phase17_feasibility_decision_gate.py#L275-L284).

### B. Full 20,000-Patient Empirical Proof
Evaluating `calculate_best_single_alarm` on the full 20,000-patient BIDMC test dataset ($1,066$ septic, $18,934$ non-septic) using $y_{\text{true}}$ ONLY yields:
- Non-septic patients ($N = 18,934$): 0 alarms issued $\implies 0.0$ achieved points.
- Septic patients with onset $t_{\text{onset}} \ge 6\text{h}$ ($N = 739$): Alarm at $t_{\text{onset}} - 6\text{h} \implies +1.0$ credit each ($739.0$ points).
- Septic patients with onset $t_{\text{onset}} < 6\text{h}$ ($N = 327$): Alarm at $t=0$, lead time $t_{\text{onset}} \implies \frac{t_{\text{onset}} + 3}{9}$ credit each ($141.777778$ points).
- Total Cohort Achieved Utility = $739.0 + 141.777778 = 880.777778$ points.
- Total Best Utility = $1066.0$ points.
- **Exact Full-Cohort Utility Ratio:** $\frac{880.777778}{1066.0} = \mathbf{+0.826245570148}$ (or **`+0.826246`**).

### C. Input & Parameter Dependencies
- **Model predictions/scores involved?** **NO** (uses $y_{\text{true}}$ only, ZERO $y_{\text{prob}}$).
- **True labels involved?** **YES** ($y_{\text{true}}$ and $t_{\text{onset}}$).
- **Thresholds/policy parameters involved?** **NO**.
- **Patient-level or cohort-level?** **Cohort-level** sum of achieved patient rewards divided by total best utility ($1066.0$).
- **Taxonomy Classification:** **`GROUND_TRUTH_ORACLE_CEILING`**

---

## 2. Number 2: `-0.234579` (`HINDSIGHT_COOLDOWN_SCORE_POLICY_CEILING`)

### A. Exact File and Function
- **Primary Generator:** [`scripts/run_m3_phase15_frozen_score_diagnostics.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/run_m3_phase15_frozen_score_diagnostics.py#L308-L318) in `best_test_oracle_u`.

### B. Verbatim Code Snippet
From [`scripts/run_m3_phase15_frozen_score_diagnostics.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/run_m3_phase15_frozen_score_diagnostics.py#L307-L318):

```python
    # Test Hindsight Threshold Sweep (DIAGNOSTIC ONLY)
    best_test_oracle_u, best_test_oracle_th = -999.0, 0.19
    test_oracle_rows = []
    for th in th_dense:
        r = evaluate_policy_fast(test_probs, test_labels, threshold=float(th), cooldown_hours=36, policy_type="cooldown")
        test_oracle_rows.append({"threshold": float(th), "utility": r["utility"], "fpr_h": r["fpr_h"], "detection": r["patient_detection"]})
        if r["utility"] > best_test_oracle_u:
            best_test_oracle_u = r["utility"]
            best_test_oracle_th = float(th)
```

### C. Input & Parameter Dependencies & Action Space
- **Model predictions/scores involved?** **YES** (sweeps frozen test probabilities `test_probs`).
- **True labels involved?** **YES** (evaluates against `test_labels` in hindsight).
- **Action Space Constraint:** **Cooldown $C = 36\text{h}$ Alert Suppression** (fires at most 1 alarm per 36h window).
- **Taxonomy Classification:** **`HINDSIGHT_COOLDOWN_SCORE_POLICY_CEILING`** (Optimal threshold $th = 0.440$, $C = 36\text{h}$).

---

## 3. Number 3: `-0.235183` (`RETRAINED_HINDSIGHT_COOLDOWN_SCORE_POLICY_CEILING`)

### A. Exact File and Function
- **Primary Generator:** [`scripts/run_m3_phase16_representation_forensics.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/run_m3_phase16_representation_forensics.py#L508-L514) in `best_exp_test_oracle_u`.

### B. Input & Parameter Dependencies & Action Space
- **Model predictions/scores involved?** **YES** (sweeps retrained Exp A / Exp F (DANN) model probabilities).
- **True labels involved?** **YES** (evaluates against `test_labels` in hindsight).
- **Action Space Constraint:** **Cooldown $C = 36\text{h}$ Alert Suppression**.
- **Taxonomy Classification:** **`RETRAINED_HINDSIGHT_COOLDOWN_SCORE_POLICY_CEILING`** (Optimal threshold $th = 0.450$, $C = 36\text{h}$). The difference from Phase 15 ($\Delta = -0.000604$) is due to slight output probability variations from retrained neural weights.

---

## 4. Number 4: `-0.855545` (`HINDSIGHT_RAW_SCORE_POLICY_CEILING`)

### A. Exact File and Function
- **Primary Generator:** [`scripts/run_m3_phase17_feasibility_decision_gate.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/run_m3_phase17_feasibility_decision_gate.py#L286-L293) in `obs_oracle_u`.

### B. Verbatim Code Snippet
From [`scripts/run_m3_phase17_feasibility_decision_gate.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/run_m3_phase17_feasibility_decision_gate.py#L286-L293):

```python
    # Hindsight Threshold Sweep WITHOUT Alert Suppression (C=0h)
    obs_oracle_u = -999.0
    for th in np.arange(0.01, 0.99, 0.005):
        preds_th = [(prs >= th).astype(int) for prs in test_probs]
        u_th = compute_utility_score(test_labels, preds_th)
        if u_th > obs_oracle_u:
            obs_oracle_u = u_th
```

### C. Input & Parameter Dependencies & Action Space Reconciliation
- **Model predictions/scores involved?** **YES** (sweeps raw test probabilities `test_probs`).
- **True labels involved?** **YES** (evaluates against `test_labels` in hindsight).
- **Action Space Constraint:** **NO Alert Suppression ($C = 0\text{h}$, Raw Instantaneous Thresholding)**. Every hour where $p(t) \ge th$ fires an alarm.
- **Why `-0.855545` differs from `-0.234579`:**
  - Without alert suppression ($C=0\text{h}$), non-septic mimic patients trigger unsuppressed false alarm hours penalized at $-0.05$ pts/hr, capping the raw threshold ceiling at **`-0.855545`** (at $th = 0.745$).
  - When Cooldown $C=36\text{h}$ alert suppression is enabled, subsequent false alarms within 36 hours are suppressed, improving the ceiling to **`-0.234579`** (at $th = 0.440$).
- **Taxonomy Classification:** **`HINDSIGHT_RAW_SCORE_POLICY_CEILING`**

---

## 📊 Complete Summary Table of Reconciled Historical Numbers

| Number | Historical Mislabeled Term | Approved Taxonomy | Involves Scores? | Involves Labels? | Action Space Constraint | Exact Empirical Source |
| :---: | :--- | :--- | :---: | :---: | :--- | :--- |
| **`+0.826246`** | "Theoretical Oracle Ceiling" | **`GROUND_TRUTH_ORACLE_CEILING`** | **NO** | **YES** | Optimal single alarm at $\max(0, t_{\text{onset}}-6)$ | Full 20,000-patient test set ($880.78 / 1066.0$) |
| **`-0.234579`** | "Phase 15 BIDMC Oracle Utility" | **`HINDSIGHT_COOLDOWN_SCORE_POLICY_CEILING`** | **YES** | **YES** | Hindsight sweep ($th=0.440$, **Cooldown $C=36\text{h}$**) | Frozen M3 probabilities on BIDMC test set |
| **`-0.235183`** | "Phase 16 BIDMC Oracle Utility" | **`RETRAINED_HINDSIGHT_COOLDOWN_SCORE_POLICY_CEILING`** | **YES** | **YES** | Hindsight sweep ($th=0.450$, **Cooldown $C=36\text{h}$**) | Retrained Exp A / Exp F probabilities |
| **`-0.855545`** | "Observable-Score Oracle Ceiling" | **`HINDSIGHT_RAW_SCORE_POLICY_CEILING`** | **YES** | **YES** | Hindsight sweep ($th=0.745$, **No Cooldown $C=0\text{h}$**) | Raw M3 probabilities without alert suppression |
