"""
run_patient_adaptive_ceiling_v2.py
-----------------------------------
Corrected Score-Based Patient-Adaptive Threshold Ceiling Calculation (V2).
Evaluates calculate_patient_adaptive_threshold_ceiling across all 20,000 BIDMC test patients.
Verifies probability-justified alarm timing, performs sanity checks, and exports:
  - results/oracle_reconciliation/patient_adaptive_ceiling_v2.csv
  - reports/oracle_reconciliation/patient_adaptive_ceiling_v2.md
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from evaluation.utility_score import compute_utility_score
from scripts.oracle_reconciliation_independent import calculate_patient_adaptive_threshold_ceiling

RESULTS_DIR = BASE_DIR / "results" / "oracle_reconciliation"
REPORTS_DIR = BASE_DIR / "reports" / "oracle_reconciliation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 95)
    print("   CORRECTED PATIENT-ADAPTIVE THRESHOLD CEILING (V2)")
    print("=" * 95)

    data = np.load(BASE_DIR / "results" / "m3_final_test_predictions.npz", allow_pickle=True)
    y_true_flat = data["y_true_flat"]
    y_proba_flat = data["y_proba_flat"]
    patient_lengths = data["patient_lengths"]

    curr = 0
    all_labels, all_probs = [], []
    for l in patient_lengths:
        all_labels.append(y_true_flat[curr : curr + l])
        all_probs.append(y_proba_flat[curr : curr + l])
        curr += l

    n_patients = len(all_labels)
    sepsis_indices = [i for i, lbls in enumerate(all_labels) if lbls.max() == 1]
    n_sepsis = len(sepsis_indices)

    print(f"Loaded {n_patients:,} test patients ({n_sepsis:,} septic).\n")

    # 1. Run Corrected Patient-Adaptive Threshold Ceiling (V2)
    norm_v2, tot_ach_v2, tot_best_v2, df_v2 = calculate_patient_adaptive_threshold_ceiling(all_labels, all_probs, cooldown_hours=72)
    df_v2.to_csv(RESULTS_DIR / "patient_adaptive_ceiling_v2.csv", index=False)
    print(f"Saved results/oracle_reconciliation/patient_adaptive_ceiling_v2.csv successfully.")

    # 2. Compare against Best Global Policy (th=0.345, C=72h)
    global_th = 0.345
    global_c = 72
    preds_global = []
    for lbls, prs in zip(all_labels, all_probs):
        p = np.zeros(len(lbls), dtype=int)
        alarm_idx = np.where(prs >= global_th)[0]
        if len(alarm_idx) > 0:
            t_curr = alarm_idx[0]
            while t_curr < len(lbls):
                if prs[t_curr] >= global_th:
                    p[t_curr] = 1
                    t_curr += global_c
                else:
                    t_curr += 1
        preds_global.append(p)

    n_diff_septic = 0
    for s_idx in sepsis_indices:
        v2_row = df_v2[df_v2["patient_id"] == s_idx].iloc[0]
        v2_first_hour = v2_row["first_alarm_hour"]
        
        g_p = preds_global[s_idx]
        g_alarm_idx = np.where(g_p == 1)[0]
        g_first_hour = int(g_alarm_idx[0]) if len(g_alarm_idx) > 0 else -1

        if v2_first_hour != g_first_hour:
            n_diff_septic += 1

    # 3. Sanity Checks
    gt_oracle_ceiling = 0.826245570148
    extended_grid_peak = -0.198307

    is_equal_gt = abs(norm_v2 - gt_oracle_ceiling) < 1e-6
    check1_pass = not is_equal_gt and (norm_v2 <= gt_oracle_ceiling + 1e-6)
    check2_pass = norm_v2 >= extended_grid_peak

    print("\n" + "=" * 95)
    print("MANDATORY VALIDATION CHECKS (V2)")
    print("=" * 95)
    print(f"PATIENT_ADAPTIVE_THRESHOLD_CEILING : {norm_v2:+.12f} ({tot_ach_v2:+.2f} / {tot_best_v2:.1f} pts)")
    print(f"Sanity Check 1 (Equal to GT Oracle?): {'YES - SUSPICIOUS (BUG PERSISTS)' if is_equal_gt else 'NO (PROBABILITY CONSTRAINED)'}")
    print(f"Sanity Check 2 (<= GT Ceiling)      : {norm_v2:+.6f} <= {gt_oracle_ceiling:+.6f} -> [{'PASS' if norm_v2 <= gt_oracle_ceiling + 1e-6 else 'FAIL'}]")
    print(f"Sanity Check 3 (>= Grid Peak)       : {norm_v2:+.6f} >= {extended_grid_peak:+.6f} -> [{'PASS' if check2_pass else 'FAIL'}]")
    print(f"Septic Patients Differing from Global Policy (th=0.345, C=72h): {n_diff_septic} / {n_sepsis} ({n_diff_septic/n_sepsis*100:.1f}%)")

    # 4. Spot-Check 3 Individual Patients (12, 54, 355)
    print("\n--- SPOT-CHECK: 3 INDIVIDUAL PATIENTS PROBABILITY TRACE ---")
    spot_ids = [12, 54, 355]
    spot_rows = []

    for sp_id in spot_ids:
        r = df_v2[df_v2["patient_id"] == sp_id].iloc[0]
        lbls = all_labels[sp_id]
        prs = all_probs[sp_id]
        t_on = r["onset_hour"]
        al_h = r["first_alarm_hour"]
        p_at_al = r["prob_at_alarm"]
        max_p = float(prs.max())
        max_p_hour = int(np.argmax(prs))

        print(f"Patient ID {sp_id:3d} | Is Sepsis: {r['is_sepsis']} | Length: {r['length_hours']}h | Onset: {t_on:2d}h")
        print(f"  Chosen Alarm Hour: {al_h:2d} | Prob at Alarm: {p_at_al:.4f} | Optimal Threshold: {r['optimal_threshold']:.4f}")
        print(f"  Patient Max Prob  : {max_p:.4f} at hour {max_p_hour:2d}")
        print(f"  Utility Contrib   : {r['optimal_utility_contribution']:+.4f}")
        print(f"  Prob Trajectory Excerpt [0:10]: {[round(float(p), 4) for p in prs[:10]]}")
        print(f"  Prob Trajectory Excerpt [onset-5:onset+3]: {[round(float(prs[t]), 4) for t in range(max(0, t_on-5), min(len(prs), t_on+3))] if t_on>=0 else []}\n")

        spot_rows.append({
            "patient_id": sp_id,
            "onset_hour": t_on,
            "first_alarm_hour": al_h,
            "prob_at_alarm": p_at_al,
            "optimal_threshold": r['optimal_threshold'],
            "utility_contrib": r['optimal_utility_contribution']
        })

    # Export markdown report
    md_content = """# 🔬 CORRECTED PATIENT-ADAPTIVE THRESHOLD CEILING ANALYSIS (V2)

## 1. Summary of Corrected Metric
The corrected `PATIENT_ADAPTIVE_THRESHOLD_CEILING` forces every alarm decision to be **PROBABILITY-JUSTIFIED**. An alarm occurs ONLY when a patient's own probability trajectory y_prob crosses candidate threshold t_i* at the first crossing hour t_alarm.

- **PATIENT_ADAPTIVE_THRESHOLD_CEILING:** `""" + f"{norm_v2:+.12f}" + """` (""" + f"{tot_ach_v2:+.2f} / {tot_best_v2:.1f}" + """ pts)
- **Ground-Truth Oracle Ceiling:** `""" + f"{gt_oracle_ceiling:+.12f}" + """`
- **Extended Grid Peak Utility (th=0.345, C=72h):** `""" + f"{extended_grid_peak:+.6f}" + """`
- **Equal to GT Oracle?** `""" + ("YES - SUSPICIOUS" if is_equal_gt else "NO (PROBABILITY-CONSTRAINED)") + """`
- **Septic Patients Differing from Global Policy:** `""" + f"{n_diff_septic}" + """` / `""" + f"{n_sepsis}" + """` (""" + f"{n_diff_septic/n_sepsis*100:.1f}" + """%)

---

## 2. Spot-Check Trace for 3 Individual Patients

### Patient 12 (Sepsis Onset at t=80)
- **Onset Hour:** 80
- **Chosen Alarm Hour:** `""" + f"{df_v2[df_v2['patient_id']==12]['first_alarm_hour'].values[0]}" + """`
- **Probability at Alarm:** `""" + f"{df_v2[df_v2['patient_id']==12]['prob_at_alarm'].values[0]:.4f}" + """`
- **Optimal Patient Threshold:** `""" + f"{df_v2[df_v2['patient_id']==12]['optimal_threshold'].values[0]:.4f}" + """`
- **Utility Contribution:** `""" + f"{df_v2[df_v2['patient_id']==12]['optimal_utility_contribution'].values[0]:+.4f}" + """`

### Patient 54 (Sepsis Onset at t=0)
- **Onset Hour:** 0
- **Chosen Alarm Hour:** `""" + f"{df_v2[df_v2['patient_id']==54]['first_alarm_hour'].values[0]}" + """`
- **Probability at Alarm:** `""" + f"{df_v2[df_v2['patient_id']==54]['prob_at_alarm'].values[0]:.4f}" + """`
- **Optimal Patient Threshold:** `""" + f"{df_v2[df_v2['patient_id']==54]['optimal_threshold'].values[0]:.4f}" + """`
- **Utility Contribution:** `""" + f"{df_v2[df_v2['patient_id']==54]['optimal_utility_contribution'].values[0]:+.4f}" + """`

### Patient 355 (Sepsis Onset at t=6)
- **Onset Hour:** 6
- **Chosen Alarm Hour:** `""" + f"{df_v2[df_v2['patient_id']==355]['first_alarm_hour'].values[0]}" + """`
- **Probability at Alarm:** `""" + f"{df_v2[df_v2['patient_id']==355]['prob_at_alarm'].values[0]:.4f}" + """`
- **Optimal Patient Threshold:** `""" + f"{df_v2[df_v2['patient_id']==355]['optimal_threshold'].values[0]:.4f}" + """`
- **Utility Contribution:** `""" + f"{df_v2[df_v2['patient_id']==355]['optimal_utility_contribution'].values[0]:+.4f}" + """`
"""
    (REPORTS_DIR / "patient_adaptive_ceiling_v2.md").write_text(md_content, encoding="utf-8")

    # 5. Print Mandatory Output Banner for Human Review
    print("=" * 95)
    print("CORRECTED PATIENT-ADAPTIVE CEILING — AWAITING HUMAN REVIEW")
    print("=" * 95)
    print(f"PATIENT_ADAPTIVE_THRESHOLD_CEILING                         : {norm_v2:+.12f}")
    print(f"Equal to GT oracle (+0.826246)?                            : {'YES - SUSPICIOUS' if is_equal_gt else 'NO'}")
    print(f">= Extended grid peak (-0.198307)                           : {'PASS' if check2_pass else 'FAIL'} ({norm_v2:+.6f} >= {extended_grid_peak:+.6f})")
    print(f"# patients where adaptive choice differs from global policy : {n_diff_septic} / {n_sepsis} ({n_diff_septic/n_sepsis*100:.1f}%)")
    print(f"RECOMMENDED CLASSIFICATION                                  : CASE B if value < 0, CASE C if value > 0 — pending human review, NOT auto-declared")
    print("=" * 95)

if __name__ == "__main__":
    main()
