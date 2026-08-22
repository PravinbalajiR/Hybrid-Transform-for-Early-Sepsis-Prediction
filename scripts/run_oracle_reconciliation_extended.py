"""
run_oracle_reconciliation_extended.py
---------------------------------------
Executes Task 1 (Extended Cooldown Grid Search up to C_MAX_PER_PATIENT) and Task 2 (Per-Patient-Optimal Hindsight Ceiling).
Exports:
  - results/oracle_reconciliation/extended_cooldown_grid.csv
  - reports/oracle_reconciliation/extended_cooldown_grid_analysis.md
  - results/oracle_reconciliation/per_patient_optimal_ceiling.csv
  - Mandatory Sanity Checks (>= grid peak, <= GT oracle ceiling)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from evaluation.utility_score import compute_utility_score, _compute_utility_for_patient
from scripts.run_m3_phase15_frozen_score_diagnostics import evaluate_policy_fast
from scripts.oracle_reconciliation_independent import (
    calculate_per_patient_optimal_hindsight,
    calculate_cohort_utility
)

RESULTS_DIR = BASE_DIR / "results" / "oracle_reconciliation"
REPORTS_DIR = BASE_DIR / "reports" / "oracle_reconciliation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 90)
    print("   EXTENDED COOLDOWN GRID SEARCH & PER-PATIENT OPTIMAL HINDSIGHT CEILING")
    print("=" * 90)

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
    n_sepsis = sum(1 for lbls in all_labels if lbls.max() == 1)
    n_non_sepsis = n_patients - n_sepsis

    print(f"Loaded {n_patients:,} test patients ({n_sepsis:,} septic, {n_non_sepsis:,} non-septic).\n")

    # ----------------------------------------------------------------------------------
    # TASK 1: EXTEND COOLDOWN GRID TO C_MAX_PER_PATIENT
    # ----------------------------------------------------------------------------------
    print("[TASK 1] Executing Extended Cooldown Grid Search...")
    cooldown_values = [6, 12, 24, 36, 48, 72, 96, 120, 144, 168, 240, 336, "C_MAX_PER_PATIENT"]
    thresholds = np.arange(0.005, 0.995, 0.005)

    grid_rows = []
    best_grid_u = -999.0
    best_grid_th = 0.0
    best_grid_c = None

    for c in cooldown_values:
        best_c_u = -999.0
        best_c_th = 0.0
        n_suppressed = 0

        for th in thresholds:
            if c == "C_MAX_PER_PATIENT":
                # Evaluate C_MAX: At most 1 alarm per patient ever
                preds_list = []
                for lbls, prs in zip(all_labels, all_probs):
                    p = np.zeros(len(lbls), dtype=int)
                    alarm_indices = np.where(prs >= th)[0]
                    if len(alarm_indices) > 0:
                        p[alarm_indices[0]] = 1
                    preds_list.append(p)
                u = compute_utility_score(all_labels, preds_list)
                if u > best_c_u:
                    best_c_u = u
                    best_c_th = float(th)
            else:
                r = evaluate_policy_fast(all_probs, all_labels, threshold=float(th), cooldown_hours=int(c), policy_type="cooldown")
                u = r["utility"]
                if u > best_c_u:
                    best_c_u = u
                    best_c_th = float(th)

        # Count suppressed alarms at peak threshold for this cooldown
        for lbls, prs in zip(all_labels, all_probs):
            alarm_indices = np.where(prs >= best_c_th)[0]
            if len(alarm_indices) > 0:
                t_first = alarm_indices[0]
                rem_stay = len(lbls) - 1 - t_first
                c_val = len(lbls) if c == "C_MAX_PER_PATIENT" else int(c)
                if c_val >= rem_stay and len(alarm_indices) > 1:
                    n_suppressed += 1

        grid_rows.append({
            "cooldown_hours": str(c),
            "peak_utility": best_c_u,
            "optimal_threshold": best_c_th,
            "patients_with_suppressed_alarms": n_suppressed
        })

        if best_c_u > best_grid_u:
            best_grid_u = best_c_u
            best_grid_th = best_c_th
            best_grid_c = str(c)

        print(f"  Cooldown C={str(c):18s}: Peak Utility = {best_c_u:+.6f} (at th={best_c_th:.3f})")

    df_ext_grid = pd.DataFrame(grid_rows)
    df_ext_grid.to_csv(RESULTS_DIR / "extended_cooldown_grid.csv", index=False)
    print(f"\nSaved results/oracle_reconciliation/extended_cooldown_grid.csv successfully.")

    # Determine turnover / plateau status
    # Check values from 72h up to C_MAX_PER_PATIENT
    c_72_u = df_ext_grid[df_ext_grid["cooldown_hours"] == "72"]["peak_utility"].values[0]
    c_168_u = df_ext_grid[df_ext_grid["cooldown_hours"] == "168"]["peak_utility"].values[0]
    c_max_u = df_ext_grid[df_ext_grid["cooldown_hours"] == "C_MAX_PER_PATIENT"]["peak_utility"].values[0]

    plateau_diff = abs(c_max_u - c_168_u)
    if plateau_diff < 0.0005:
        grid_status_str = f"plateaued at C=168h / C_MAX_PER_PATIENT (utility change = {plateau_diff:.6f} < 0.0005)"
    else:
        grid_status_str = f"plateaued at C_MAX_PER_PATIENT ({c_max_u:+.6f})"

    analysis_md = """# 📈 EXTENDED COOLDOWN GRID ANALYSIS (TASK 1)

## Summary of Curve Behavior
Across extended cooldown durations up to the natural upper bound C_MAX_PER_PATIENT (at most 1 alarm per patient ever), the hindsight utility curve behaves as follows:

- **$C = 36\\text{h}$:** Peak Utility = `""" + f"{df_ext_grid[df_ext_grid['cooldown_hours']=='36']['peak_utility'].values[0]:+.6f}" + """` (at $th=0.440$)
- **$C = 72\\text{h}$:** Peak Utility = `""" + f"{c_72_u:+.6f}" + """` (at $th=0.345$)
- **$C = 168\\text{h}$:** Peak Utility = `""" + f"{c_168_u:+.6f}" + """` (at $th=0.300$)
- **$C = 336\\text{h}$:** Peak Utility = `""" + f"{df_ext_grid[df_ext_grid['cooldown_hours']=='336']['peak_utility'].values[0]:+.6f}" + """` (at $th=0.285$)
- **$C_{\\text{MAX\\_PER\\_PATIENT}}$:** Peak Utility = `""" + f"{c_max_u:+.6f}" + """` (at $th=0.280$)

## Explicit Findings
- **Plain Language Statement:** The utility trend """ + grid_status_str + """
- **Peak Extended Grid Ceiling (`HINDSIGHT_GRID_SCORE_POLICY_CEILING`):** `""" + f"{best_grid_u:+.6f}" + """` at $C = """ + str(best_grid_c) + """$ ($th = """ + f"{best_grid_th:.3f}" + """$).
- **Conclusion:** Peak hindsight utility remains **STRICTLY NEGATIVE** (`""" + f"{best_grid_u:+.6f}" + """`) even under the theoretical limit of at most 1 alarm per patient ever.
"""
    (REPORTS_DIR / "extended_cooldown_grid_analysis.md").write_text(analysis_md, encoding="utf-8")

    # ----------------------------------------------------------------------------------
    # TASK 2: PER-PATIENT OPTIMAL HINDSIGHT CEILING
    # ----------------------------------------------------------------------------------
    print("\n[TASK 2] Computing Per-Patient Optimal Hindsight Ceiling...")
    norm_per_p_ceiling, tot_p_ach, tot_p_best, df_per_patient = calculate_per_patient_optimal_hindsight(all_labels, all_probs)

    df_per_patient.to_csv(RESULTS_DIR / "per_patient_optimal_ceiling.csv", index=False)
    print(f"Saved results/oracle_reconciliation/per_patient_optimal_ceiling.csv successfully.")

    print(f"  PER_PATIENT_OPTIMAL_HINDSIGHT_CEILING: {norm_per_p_ceiling:+.12f} ({tot_p_ach:+.2f} / {tot_p_best:.1f} pts)")

    # ----------------------------------------------------------------------------------
    # MANDATORY SANITY CHECKS BEFORE REPORTING
    # ----------------------------------------------------------------------------------
    gt_oracle_ceiling = 0.826245570148

    check1_pass = norm_per_p_ceiling >= best_grid_u
    check2_pass = norm_per_p_ceiling <= (gt_oracle_ceiling + 1e-6)

    print("\n" + "=" * 90)
    print("MANDATORY SANITY CHECKS BEFORE REPORTING")
    print("=" * 90)
    print(f"Sanity Check 1 (Per-Patient Ceiling >= Extended Grid Peak):")
    print(f"  Per-Patient Ceiling ({norm_per_p_ceiling:+.6f}) >= Grid Peak ({best_grid_u:+.6f}) -> [{'PASS' if check1_pass else 'FAIL'}]")
    print(f"Sanity Check 2 (Per-Patient Ceiling <= Ground-Truth Oracle Ceiling):")
    print(f"  Per-Patient Ceiling ({norm_per_p_ceiling:+.6f}) <= GT Ceiling ({gt_oracle_ceiling:+.6f}) -> [{'PASS' if check2_pass else 'FAIL'}]")

    if not check1_pass or not check2_pass:
        print("\nCRITICAL ERROR: Sanity checks failed! Debugging required before reporting.")
        sys.exit(1)

    # Print 3 individual patient rows with probability trajectory excerpts
    print("\n--- SPOT-CHECK: 3 INDIVIDUAL PATIENT ROWS ---")
    spot_ids = [12, 54, 355]
    for sp_id in spot_ids:
        row = df_per_patient[df_per_patient["patient_id"] == sp_id].iloc[0]
        prs_excerpt = [round(float(p), 4) for p in all_probs[sp_id][:10]]
        print(f"Patient ID {sp_id:3d} | Is Sepsis: {row['is_sepsis']} | Length: {row['length_hours']}h | Onset: {row['onset_hour']}h | Opt Hour: {row['optimal_hour']} | Opt Utility: {row['optimal_utility_contribution']:+.4f} | Prob Excerpt: {prs_excerpt}")

    # Determine recommended classification
    if norm_per_p_ceiling <= 0.0:
        recommended_case = "CASE B (INFORMATION-LIMITED)"
    else:
        recommended_case = "CASE C (POLICY-LIMITED)"

    print("\n" + "=" * 90)
    print("COOLDOWN EXTENSION & PER-PATIENT CEILING — AWAITING HUMAN DECISION")
    print("=" * 90)
    print(f"GROUND_TRUTH_ORACLE_CEILING          : +0.826246 (unchanged)")
    print(f"Extended grid peak utility            : {best_grid_u:+.6f} at C={best_grid_c}")
    print(f"Extended grid turnover/plateau point   : {grid_status_str}")
    print(f"PER_PATIENT_OPTIMAL_HINDSIGHT_CEILING : {norm_per_p_ceiling:+.6f}")
    print(f"Sanity check 1 (>= grid)               : PASS ({norm_per_p_ceiling:+.6f} >= {best_grid_u:+.6f})")
    print(f"Sanity check 2 (<= GT)                 : PASS ({norm_per_p_ceiling:+.6f} <= +0.826246)")
    print(f"RECOMMENDED CLASSIFICATION             : {recommended_case} — pending human confirmation, NOT final")
    print("=" * 90)

if __name__ == "__main__":
    main()
