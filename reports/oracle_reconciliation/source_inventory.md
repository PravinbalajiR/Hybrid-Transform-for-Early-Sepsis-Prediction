# 🔍 SOURCE INVENTORY FOR HISTORICAL ORACLE / CEILING NUMBERS

This document establishes the exact code origin, calculation mechanism, input dependencies, and normalization scheme for the four historical numbers reported across Phases 14–17 of the `Hybrid-Transform-for-Early-Sepsis-Prediction` project.

---

## 1. Number 1: `+0.826246` ("Theoretical Oracle Utility Ceiling" / "Ground-Truth Oracle Utility")

### A. Exact File and Function
- **Primary Generator:** [`scripts/recompute_exact_decompositions.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/recompute_exact_decompositions.py#L180-L208) in `run_threshold_decomposition()`.
- **Primary Hardcoded Reference:** [`scripts/run_m3_phase14_utr.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/run_m3_phase14_utr.py#L480-L488) in `envelope_rows`.
- **Secondary References:** [`scripts/run_m3_phase16_representation_forensics.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/run_m3_phase16_representation_forensics.py#L677) (hardcoded string) and [`scripts/run_m3_phase17_feasibility_decision_gate.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/run_m3_phase17_feasibility_decision_gate.py#L275-L284) (`u_gt_oracle`).

### B. Verbatim Code Snippet
From [`scripts/run_m3_phase14_utr.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/run_m3_phase14_utr.py#L473-L488):

```python
    # Level 2: Oracle Threshold (fast numpy sweep)
    best_th_oracle, max_u_oracle = best_val_th_selected, -999.0
    for th in np.arange(0.05, 0.95, 0.02):
        r_o = evaluate_probs_list(best_val_probs_test, test_labels, threshold=float(th), cooldown_hours=36, policy_name="Oracle")
        if r_o["utility"] > max_u_oracle:
            max_u_oracle = r_o["utility"]
            best_th_oracle = float(th)

    envelope_rows = [
        {"Level": "1. Current Predictions (Frozen Val Policy)", "Test_Utility": float(res_raw["utility"]), "Description": "Single-pass zero-leakage evaluation"},
        {"Level": "2. Oracle Threshold (Diagnostic Only)", "Test_Utility": float(max_u_oracle), "Description": "Best test utility under optimal test threshold"},
        {"Level": "3. Oracle Temporal Cooldown Policy", "Test_Utility": float(max_u_oracle + 0.005), "Description": "Optimal alert suppression policy per patient"},
        {"Level": "4. Oracle Ranking (Perfect Separability)", "Test_Utility": +0.826246, "Description": "Theoretical upper bound on existing predictions"},
        {"Level": "5. Perfect Label Oracle", "Test_Utility": +1.000000, "Description": "100% TP reward with zero false alarm penalty"},
    ]
```

From [`scripts/recompute_exact_decompositions.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/recompute_exact_decompositions.py#L180-L203) (where `0.826246` is computed as `sum_tp_reward / 1066.0`):

```python
    thresholds = [0.44, 0.60, 0.78]
    results = []

    for th in thresholds:
        res = run_threshold_decomposition(all_labels, all_probs, th)
        results.append(res)

        print(f"--- DECOMPOSITION AT THRESHOLD {th:.2f} ---")
        print(f"  Early-Warning TP Reward      : +{res['sum_tp_reward']:.2f} pts")
        print(f"  Total Best Possible Utility  : {res['total_best_utility']:.2f} pts")
        print(f"  NORMALIZED PHYSIONET UTILITY : {res['normalized_utility']:+.4f}")
```

### C. Input & Parameter Dependencies
- **Model predictions/scores involved?**
  - In `Phase 14` / `recompute_exact_decompositions.py`: **YES** (evaluating `y_proba_flat` at $th=0.78$ where `sum_tp_reward` = `880.778278` and total best utility = `1066.0`, yielding $\frac{880.778278}{1066.0} = 0.826246$).
  - In `Phase 17` (`u_gt_oracle`): **NO** (evaluating ground-truth optimal single alarm timing $\max(0, t_{onset} - 6)$ per septic patient using true labels only).
- **True labels involved?** **YES** (line 470 in `run_m3_phase14_utr.py`, line 106 in `recompute_exact_decompositions.py`).
- **Thresholds/policy parameters involved?** **YES** ($th=0.78$ in `recompute_exact_decompositions.py`, line 180).
- **Patient-level or cohort-level?** **Cohort-level** sum of patient achieved rewards divided by total best utility.
- **Normalized?** **YES** (divided by $N_{\text{septic}} \times 1.0 = 1066.0$).

---

## 2. Number 2: `-0.234579` ("Phase 15 BIDMC Oracle Utility")

### A. Exact File and Function
- **Primary Generator:** [`scripts/run_m3_phase15_frozen_score_diagnostics.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/run_m3_phase15_frozen_score_diagnostics.py#L308-L318) in `best_test_oracle_u`.
- **Secondary References:** [`scripts/run_m3_phase16_representation_forensics.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/run_m3_phase16_representation_forensics.py#L609) (`phase15_oracle_baseline`) and [`scripts/run_m3_phase17_feasibility_decision_gate.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/run_m3_phase17_feasibility_decision_gate.py#L517).

### B. Verbatim Code Snippet
From [`scripts/run_m3_phase15_frozen_score_diagnostics.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/run_m3_phase15_frozen_score_diagnostics.py#L307-L318):

```python
    # Test Oracle Threshold Sweep (DIAGNOSTIC ONLY)
    best_test_oracle_u, best_test_oracle_th = -999.0, 0.19
    test_oracle_rows = []
    for th in th_dense:
        r = evaluate_policy_fast(test_probs, test_labels, threshold=float(th), cooldown_hours=36, policy_type="cooldown")
        test_oracle_rows.append({"threshold": float(th), "utility": r["utility"], "fpr_h": r["fpr_h"], "detection": r["patient_detection"]})
        if r["utility"] > best_test_oracle_u:
            best_test_oracle_u = r["utility"]
            best_test_oracle_th = float(th)

    df_test_oracle = pd.DataFrame(test_oracle_rows)
    save_dual(df_test_oracle, "phase15_test_oracle_threshold_frontier.csv")
```

### C. Input & Parameter Dependencies
- **Model predictions/scores involved?** **YES** (evaluates frozen test probabilities `test_probs` from `m3_final_test_predictions.npz`, line 311).
- **True labels involved?** **YES** (evaluates against `test_labels`, line 311).
- **Thresholds/policy parameters involved?** **YES** (sweeps `th_dense = np.arange(0.005, 0.995, 0.005)` with `cooldown_hours=36`, line 292, 311).
- **Patient-level or cohort-level?** **Cohort-level** normalized utility.
- **Normalized?** **YES** (divided by $N_{\text{septic}} \times 1.0 = 1066.0$).

---

## 3. Number 3: `-0.235183` ("Phase 16 BIDMC Oracle Utility")

### A. Exact File and Function
- **Primary Generator:** [`scripts/run_m3_phase16_representation_forensics.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/run_m3_phase16_representation_forensics.py#L508-L514) in `best_exp_test_oracle_u` for Retrained Exp A.
- **Secondary References:** [`results/phase16/phase16_ablation.csv`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/results/phase16/phase16_ablation.csv) and `results/phase16/phase16_diagnostic_summary.json`.

### B. Verbatim Code Snippet
From [`scripts/run_m3_phase16_representation_forensics.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/run_m3_phase16_representation_forensics.py#L507-L518):

```python
        # Compute Diagnostic Test Oracle Threshold for this model
        best_exp_test_oracle_u = -999.0
        for th_cand in np.arange(0.05, 0.95, 0.02):
            rt_cand = evaluate_probs_list(p_test_list if exp_id != "A" else test_probs, test_labels, threshold=float(th_cand), cooldown_hours=36, policy_name="TestOracle")
            if rt_cand["utility"] > best_exp_test_oracle_u:
                best_exp_test_oracle_u = rt_cand["utility"]

        if res_v["utility"] > best_val_u:
            best_val_u = res_v["utility"]
            best_val_exp_id = exp_id

        if best_exp_test_oracle_u > best_oracle_test_u:
            best_oracle_test_u = best_exp_test_oracle_u
```

### C. Input & Parameter Dependencies
- **Model predictions/scores involved?** **YES** (evaluates retrained Exp A test probabilities `p_test_list`, line 510).
- **True labels involved?** **YES** (evaluates against `test_labels`, line 510).
- **Thresholds/policy parameters involved?** **YES** (sweeps `th_cand` from $0.05$ to $0.95$ at $0.02$ step with `cooldown_hours=36`, line 509, 510).
- **Patient-level or cohort-level?** **Cohort-level** normalized utility.
- **Normalized?** **YES** (divided by $N_{\text{septic}} \times 1.0 = 1066.0$).

---

## 4. Number 4: `-0.855545` ("Observable-Score Oracle Ceiling")

### A. Exact File and Function
- **Primary Generator:** [`scripts/run_m3_phase17_feasibility_decision_gate.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/run_m3_phase17_feasibility_decision_gate.py#L286-L293) in `obs_oracle_u`.
- **Secondary References:** `results/phase17_score_separability.csv` and `results/phase17_decision_gate.json`.

### B. Verbatim Code Snippet
From [`scripts/run_m3_phase17_feasibility_decision_gate.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/run_m3_phase17_feasibility_decision_gate.py#L286-L298):

```python
    # Observable-Score Threshold Sweep Oracle (Phase 15/16 baseline)
    obs_oracle_u = -999.0
    for th in np.arange(0.01, 0.99, 0.005):
        preds_th = [(prs >= th).astype(int) for prs in test_probs]
        u_th = compute_utility_score(test_labels, preds_th)
        if u_th > obs_oracle_u:
            obs_oracle_u = u_th

    print_flush(f"   A. Never Alarm Utility           : {u_never:+.6f}")
    print_flush(f"   B. Always Alarm Utility          : {u_always:+.6f}")
    print_flush(f"   C. Onset Alarm Utility           : {u_onset:+.6f}")
    print_flush(f"   D. Ground-Truth Oracle Ceiling   : {u_gt_oracle:+.6f} [MAX_BIDMC_ORACLE_UTILITY]")
    print_flush(f"   E. Observable-Score Oracle Ceiling: {obs_oracle_u:+.6f} (Phase 15 Baseline)")
```

### C. Input & Parameter Dependencies
- **Model predictions/scores involved?** **YES** (evaluates raw test probabilities `test_probs` from `m3_final_test_predictions.npz`, line 289).
- **True labels involved?** **YES** (evaluates against `test_labels`, line 290).
- **Thresholds/policy parameters involved?** **YES** (sweeps threshold $th \in [0.01, 0.99]$ at $0.005$ resolution with NO cooldown policy, $C=0$).
- **Patient-level or cohort-level?** **Cohort-level** normalized utility.
- **Normalized?** **YES** (divided by $N_{\text{septic}} \times 1.0 = 1066.0$).

---

## 📊 Summary Table of Historical Numbers

| Number | Historical Label | Involves Scores? | Involves Labels? | Involves Thresholds/Policies? | Actual Classification |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **`+0.826246`** | "Theoretical Oracle Utility Ceiling" | NO (in Phase 17 GT Oracle) / YES (in Phase 14) | YES | YES (in Phase 14) / NO (in Phase 17 GT Oracle) | **Ground-Truth Perfect-Information Oracle Ceiling** (or TP reward ratio at $th=0.78$) |
| **`-0.234579`** | "Phase 15 BIDMC Oracle Utility" | YES | YES | YES ($th \in [0.005, 0.995], C=36\text{h}$) | **Score-Based Policy Ceiling** (Frozen M3 predictions + Cooldown 36h) |
| **`-0.235183`** | "Phase 16 BIDMC Oracle Utility" | YES | YES | YES ($th \in [0.05, 0.95], C=36\text{h}$) | **Score-Based Policy Ceiling** (Retrained Exp A predictions + Cooldown 36h) |
| **`-0.855545`** | "Observable-Score Oracle Ceiling" | YES | YES | YES ($th \in [0.01, 0.99], C=0\text{h}$) | **Score-Based Threshold Ceiling** (Raw probabilities without alert suppression) |
