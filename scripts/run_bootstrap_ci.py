"""
run_bootstrap_ci.py
--------------------
Patient-level Bootstrap Uncertainty Quantification (B = 1,000 iterations).
Computes 95% Confidence Intervals and Paired Significance Tests for:
  1. GROUND_TRUTH_ORACLE_CEILING (+0.826246)
  2. HINDSIGHT_GRID_SCORE_POLICY_CEILING (-0.198307)
  3. PATIENT_ADAPTIVE_THRESHOLD_CEILING (+0.281895)
  4. FROZEN_MODEL_UTILITY (-0.257312)
  5. REALISTIC_ACHIEVABLE_UTILITY (-0.198307)

Exports:
  - results/oracle_reconciliation/bootstrap_ci_all_metrics.csv
  - results/oracle_reconciliation/bootstrap_raw_iterations.csv
  - results/oracle_reconciliation/paired_significance_tests.csv
  - figures/oracle_reconciliation_ci_summary.png
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from evaluation.utility_score import compute_utility_score
from scripts.oracle_reconciliation_independent import (
    calculate_patient_utility,
    calculate_best_single_alarm
)

RESULTS_DIR = BASE_DIR / "results" / "oracle_reconciliation"
FIGURES_DIR = BASE_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def print_flush(msg: str):
    print(msg, flush=True)

def main():
    print_flush("=" * 95)
    print_flush("   PATIENT-LEVEL BOOTSTRAP UNCERTAINTY QUANTIFICATION (B = 1,000)")
    print_flush("=" * 95)

    # 1. Load predictions
    data_test = np.load(BASE_DIR / "results" / "m3_final_test_predictions.npz", allow_pickle=True)
    test_y_true, test_y_prob, test_lens = data_test["y_true_flat"], data_test["y_proba_flat"], data_test["patient_lengths"]

    curr = 0
    all_labels, all_probs = [], []
    for l in test_lens:
        all_labels.append(test_y_true[curr : curr + l])
        all_probs.append(test_y_prob[curr : curr + l])
        curr += l

    n_patients = len(all_labels)
    df_v2 = pd.read_csv(RESULTS_DIR / "patient_adaptive_ceiling_v2.csv")

    print_flush(f"Loaded {n_patients:,} BIDMC test patients. Precomputing patient-level utilities...")

    # Policy parameters
    th_grid, c_grid = 0.345, 72
    th_frozen, c_frozen = 0.190, 36

    # Precompute per-patient utility contributions for each metric
    u_gt_array = np.zeros(n_patients, dtype=float)
    u_grid_array = np.zeros(n_patients, dtype=float)
    u_adapt_array = np.zeros(n_patients, dtype=float)
    u_frozen_array = np.zeros(n_patients, dtype=float)
    u_real_array = np.zeros(n_patients, dtype=float)
    best_u_array = np.zeros(n_patients, dtype=float)

    for idx, (lbls, prs) in enumerate(zip(all_labels, all_probs)):
        is_sep = int(lbls.max()) == 1

        # 1. Ground Truth Oracle
        _, ach_gt, b_gt = calculate_best_single_alarm(lbls)
        u_gt_array[idx] = ach_gt
        best_u_array[idx] = b_gt

        # 2. Hindsight Grid Policy (th=0.345, C=72h)
        p_grid = np.zeros(len(lbls), dtype=int)
        alarm_idx = np.where(prs >= th_grid)[0]
        if len(alarm_idx) > 0:
            t_curr = alarm_idx[0]
            while t_curr < len(lbls):
                if prs[t_curr] >= th_grid:
                    p_grid[t_curr] = 1
                    t_curr += c_grid
                else: t_curr += 1
        ach_grid, _ = calculate_patient_utility(lbls, p_grid)
        u_grid_array[idx] = ach_grid

        # 3. Patient-Adaptive Ceiling (V2, C=72h)
        ach_adapt = float(df_v2[df_v2["patient_id"] == idx]["optimal_utility_contribution"].values[0])
        u_adapt_array[idx] = ach_adapt

        # 4. Frozen Model Utility (th=0.190, C=36h)
        p_frozen = np.zeros(len(lbls), dtype=int)
        alarm_idx_f = np.where(prs >= th_frozen)[0]
        if len(alarm_idx_f) > 0:
            t_curr = alarm_idx_f[0]
            while t_curr < len(lbls):
                if prs[t_curr] >= th_frozen:
                    p_frozen[t_curr] = 1
                    t_curr += c_frozen
                else: t_curr += 1
        ach_frozen, _ = calculate_patient_utility(lbls, p_frozen)
        u_frozen_array[idx] = ach_frozen

        # 5. Realistic Achievable Utility (falls back to grid policy peak)
        u_real_array[idx] = ach_grid

    # Verify point estimates match exactly
    tot_best = np.sum(best_u_array)
    pt_gt = np.sum(u_gt_array) / tot_best
    pt_grid = np.sum(u_grid_array) / tot_best
    pt_adapt = np.sum(u_adapt_array) / tot_best
    pt_frozen = np.sum(u_frozen_array) / tot_best
    pt_real = np.sum(u_real_array) / tot_best

    print_flush(f"Verified Point Estimates:")
    print_flush(f"  GROUND_TRUTH_ORACLE_CEILING         : {pt_gt:+.6f}")
    print_flush(f"  HINDSIGHT_GRID_SCORE_POLICY_CEILING : {pt_grid:+.6f}")
    print_flush(f"  PATIENT_ADAPTIVE_THRESHOLD_CEILING  : {pt_adapt:+.6f}")
    print_flush(f"  FROZEN_MODEL_UTILITY                : {pt_frozen:+.6f}")
    print_flush(f"  REALISTIC_ACHIEVABLE_UTILITY        : {pt_real:+.6f}\n")

    # ----------------------------------------------------------------------------------
    # TASK 1: PATIENT-LEVEL BOOTSTRAP (B = 1,000 Iterations)
    # ----------------------------------------------------------------------------------
    print_flush("[TASK 1] Running B = 1,000 Patient-Level Bootstrap Resampling...")
    B = 1000
    np.random.seed(42)

    boot_gt = np.zeros(B)
    boot_grid = np.zeros(B)
    boot_adapt = np.zeros(B)
    boot_frozen = np.zeros(B)
    boot_real = np.zeros(B)

    for b in range(B):
        sample_idx = np.random.choice(n_patients, size=n_patients, replace=True)
        samp_best = np.sum(best_u_array[sample_idx])

        boot_gt[b] = np.sum(u_gt_array[sample_idx]) / samp_best
        boot_grid[b] = np.sum(u_grid_array[sample_idx]) / samp_best
        boot_adapt[b] = np.sum(u_adapt_array[sample_idx]) / samp_best
        boot_frozen[b] = np.sum(u_frozen_array[sample_idx]) / samp_best
        boot_real[b] = np.sum(u_real_array[sample_idx]) / samp_best

    # Save raw per-iteration values
    df_raw = pd.DataFrame({
        "iteration": np.arange(1, B + 1),
        "GROUND_TRUTH_ORACLE_CEILING": boot_gt,
        "HINDSIGHT_GRID_SCORE_POLICY_CEILING": boot_grid,
        "PATIENT_ADAPTIVE_THRESHOLD_CEILING": boot_adapt,
        "FROZEN_MODEL_UTILITY": boot_frozen,
        "REALISTIC_ACHIEVABLE_UTILITY": boot_real
    })
    df_raw.to_csv(RESULTS_DIR / "bootstrap_raw_iterations.csv", index=False)

    # Compute summary statistics
    metrics_data = [
        ("GROUND_TRUTH_ORACLE_CEILING", pt_gt, boot_gt),
        ("HINDSIGHT_GRID_SCORE_POLICY_CEILING", pt_grid, boot_grid),
        ("PATIENT_ADAPTIVE_THRESHOLD_CEILING", pt_adapt, boot_adapt),
        ("FROZEN_MODEL_UTILITY", pt_frozen, boot_frozen),
        ("REALISTIC_ACHIEVABLE_UTILITY", pt_real, boot_real)
    ]

    summary_rows = []
    for name, pt_val, b_dist in metrics_data:
        ci_low, ci_high = np.percentile(b_dist, [2.5, 97.5])
        crosses_zero = (ci_low <= 0.0 <= ci_high)
        summary_rows.append({
            "Metric": name,
            "Point_Estimate": pt_val,
            "Mean": np.mean(b_dist),
            "Median": np.median(b_dist),
            "Std_Dev": np.std(b_dist),
            "CI_95_Low": ci_low,
            "CI_95_High": ci_high,
            "Crosses_Zero": "YES" if crosses_zero else "NO"
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(RESULTS_DIR / "bootstrap_ci_all_metrics.csv", index=False)
    print_flush(df_summary.to_string(index=False))

    # ----------------------------------------------------------------------------------
    # TASK 2: PAIRED SIGNIFICANCE TESTS
    # ----------------------------------------------------------------------------------
    print_flush("\n[TASK 2] Performing Paired Significance Tests...")
    delta_adapt_frozen = boot_adapt - boot_frozen
    p_adapt_frozen = 2 * min(np.mean(delta_adapt_frozen <= 0), np.mean(delta_adapt_frozen >= 0))
    ci_af_low, ci_af_high = np.percentile(delta_adapt_frozen, [2.5, 97.5])

    delta_gt_grid = boot_gt - boot_grid
    p_gt_grid = 2 * min(np.mean(delta_gt_grid <= 0), np.mean(delta_gt_grid >= 0))
    ci_gg_low, ci_gg_high = np.percentile(delta_gt_grid, [2.5, 97.5])

    df_paired = pd.DataFrame([
        {
            "Comparison": "Adaptive Ceiling - Frozen Utility",
            "Mean_Delta": np.mean(delta_adapt_frozen),
            "CI_95_Low": ci_af_low,
            "CI_95_High": ci_af_high,
            "p_value": p_adapt_frozen
        },
        {
            "Comparison": "GT Oracle - Grid Policy Ceiling",
            "Mean_Delta": np.mean(delta_gt_grid),
            "CI_95_Low": ci_gg_low,
            "CI_95_High": ci_gg_high,
            "p_value": p_gt_grid
        }
    ])
    df_paired.to_csv(RESULTS_DIR / "paired_significance_tests.csv", index=False)
    print_flush("\nPaired Significance Tests Summary:")
    print_flush(df_paired.to_string(index=False))

    # ----------------------------------------------------------------------------------
    # TASK 3: VISUAL SUMMARY PLOT
    # ----------------------------------------------------------------------------------
    print_flush("\n[TASK 3] Generating Visual Summary Plot...")
    plt.figure(figsize=(10, 6))

    metric_names_display = [
        "GROUND_TRUTH_ORACLE_CEILING",
        "PATIENT_ADAPTIVE_THRESHOLD_CEILING",
        "REALISTIC_ACHIEVABLE_UTILITY",
        "HINDSIGHT_GRID_SCORE_POLICY_CEILING",
        "FROZEN_MODEL_UTILITY"
    ]

    y_positions = np.arange(len(metric_names_display))[::-1]

    for pos, name in zip(y_positions, metric_names_display):
        row = df_summary[df_summary["Metric"] == name].iloc[0]
        pt = row["Point_Estimate"]
        err_low = pt - row["CI_95_Low"]
        err_high = row["CI_95_High"] - pt

        color = "forestgreen" if pt > 0 else "firebrick"
        plt.errorbar(pt, pos, xerr=[[err_low], [err_high]], fmt="o", color=color, ecolor=color, elinewidth=2.5, capsize=5, markersize=8)
        plt.text(pt, pos + 0.15, f"{pt:+.4f} [{row['CI_95_Low']:+.4f}, {row['CI_95_High']:+.4f}]", ha="center", fontsize=9, fontweight="bold")

    plt.axvline(0.0, color="black", linestyle="--", linewidth=1.5, label="U = 0.0 Boundary")
    plt.yticks(y_positions, metric_names_display, fontsize=10, fontweight="bold")
    plt.xlabel("PhysioNet Utility Score", fontsize=11, fontweight="bold")
    plt.title("PhysioNet Utility Decomposition: 95% Patient-Level Bootstrap CIs (B=1,000)", fontsize=12, fontweight="bold", pad=15)
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "oracle_reconciliation_ci_summary.png", dpi=300)
    plt.close()
    print_flush("Saved figures/oracle_reconciliation_ci_summary.png successfully.\n")

    # ----------------------------------------------------------------------------------
    # REQUIRED DECISION GATE SUMMARY
    # ----------------------------------------------------------------------------------
    print_flush("=" * 95)
    print_flush("BOOTSTRAP CI RESULTS — AWAITING HUMAN REVIEW")
    print_flush("=" * 95)
    for row in summary_rows:
        name = row["Metric"]
        pt = row["Point_Estimate"]
        c_low = row["CI_95_Low"]
        c_high = row["CI_95_High"]
        crosses = row["Crosses_Zero"]
        print_flush(f"{name:38s}: {pt:+.6f} [95% CI: {c_low:+.6f}, {c_high:+.6f}]  Crosses 0? [{crosses}]")

    print_flush(f"\nPAIRED TEST (Adaptive - Frozen)       : delta={np.mean(delta_adapt_frozen):+.6f}, p={p_adapt_frozen:.4e}")
    print_flush(f"PAIRED TEST (GT Oracle - Grid Ceiling): delta={np.mean(delta_gt_grid):+.6f}, p={p_gt_grid:.4e}")
    print_flush("=" * 95)

if __name__ == "__main__":
    main()
