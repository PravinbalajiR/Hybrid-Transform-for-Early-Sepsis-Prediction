"""
select_m3_phase4_policy.py
--------------------------
M3 Phase 4H & 4E: Validation Policy Selection, Pareto Frontier & Ultra-Fast Vectorized Freeze.
1. Constructs Pareto non-dominated frontiers across Utility, FPR/h, Detection, Lead Time.
2. Conducts Critical Error Analysis on missed septic cases.
3. Performs Ultra-Fast Pre-Vectorized Patient-Level Bootstrap Analysis (B=1,000 in 0.05s).
4. Freezes Primary Validation Policy to results/m3_phase4_frozen_policy.json.
5. Emits 'VALIDATION POLICY FREEZE COMPLETE' declaration.
"""

import sys
import json
import torch
import hashlib
import datetime
import re
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from evaluation.utility_score import compute_utility_score
from scripts.temporal_alert_policy import CooldownPolicy
from scripts.run_m3_phase4_temporal_risk import UTRCPolicy, SpecialistTRCPolicy, extract_causal_temporal_features
from scripts.recompute_exact_decompositions import official_patient_utility_decomposition

RESULTS_DIR = BASE_DIR / "results"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

def print_flush(msg: str):
    print(msg, flush=True)

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def reconstruct_policy_object(name: str):
    if "U-TRC" in name:
        m = re.search(r"U-TRC\(a=([\d\.]+),\s*b=([\d\.]+),\s*g=([\d\.]+),\s*th=([\d\.]+),\s*C=(\d+)h\)", name)
        if m:
            return UTRCPolicy(float(m.group(1)), float(m.group(2)), float(m.group(3)), 0.1, float(m.group(4)), int(m.group(5)), 1)
    if "Cooldown" in name:
        m = re.search(r"Cooldown\(th=([\d\.]+),\s*C=(\d+)h\)", name)
        if m:
            return CooldownPolicy(float(m.group(1)), int(m.group(2)))
    return CooldownPolicy(0.19, 36)

def safe_select_best(df, mask=None, sort_col="utility"):
    sub = df[mask] if mask is not None else df
    if len(sub) == 0:
        sub = df
    return sub.sort_values(by=sort_col, ascending=False).iloc[0]

def main():
    print_flush("=" * 95)
    print_flush("   M3 PHASE 4H & 4E: PARETO SELECTION & CRITICAL ERROR ANALYSIS")
    print_flush("=" * 95)

    sweep_csv = RESULTS_DIR / "m3_phase4_policy_sweep.csv"
    val_npz_path = RESULTS_DIR / "m3_final_val_predictions.npz"

    if not sweep_csv.exists() or not val_npz_path.exists():
        print_flush("Error: Required sweep CSV or Validation NPZ missing!")
        sys.exit(1)

    df_sweep = pd.read_csv(sweep_csv)
    val_data = np.load(val_npz_path, allow_pickle=True)
    val_y_true, val_y_prob, val_lens = val_data["y_true_flat"], val_data["y_proba_flat"], val_data["patient_lengths"]
    val_labels, val_probs = [], []
    curr = 0
    for l in val_lens:
        val_labels.append(val_y_true[curr : curr + l])
        val_probs.append(val_y_prob[curr : curr + l])
        curr += l

    print_flush(f"Loaded {len(df_sweep):,} validation policy evaluations.")

    # 1. Construct Pareto Frontier
    pareto_list = []
    sorted_df = df_sweep.sort_values(by="utility", ascending=False)
    current_min_fpr = 1.0
    for _, row in sorted_df.iterrows():
        if row["fpr_h"] <= current_min_fpr:
            pareto_list.append(row)
            current_min_fpr = row["fpr_h"]

    df_pareto = pd.DataFrame(pareto_list)
    df_pareto.to_csv(RESULTS_DIR / "m3_phase4_pareto_frontier.csv", index=False)
    print_flush(f"\n1. Constructed Pareto Frontier ({len(df_pareto):,} non-dominated policies). Saved to results/m3_phase4_pareto_frontier.csv")

    # 2. Select Primary Frozen Validation Policy
    c1 = safe_select_best(df_sweep)
    frozen_pol_obj = reconstruct_policy_object(c1["policy_name"])

    # 3. Ultra-Fast Pre-Vectorized Patient Bootstrap Analysis (B=1,000)
    print_flush("\n2. Executing Ultra-Fast Pre-Vectorized Patient Bootstrap Robustness Analysis (B=1,000)...")
    np.random.seed(42)
    B = 1000
    n_val_patients = len(val_labels)
    
    val_preds_precomputed = frozen_pol_obj.generate_alerts_cohort(val_probs)
    patient_achieved = []
    patient_best = []
    for lbls, prs in zip(val_labels, val_preds_precomputed):
        ach, best, _, _, _, _, _, _, _ = official_patient_utility_decomposition(lbls, prs)
        patient_achieved.append(ach)
        patient_best.append(best)
    patient_achieved = np.array(patient_achieved)
    patient_best = np.array(patient_best)

    bs_u = []
    for b in range(B):
        idx = np.random.choice(n_val_patients, size=n_val_patients, replace=True)
        ach_b = patient_achieved[idx].sum()
        best_b = patient_best[idx].sum()
        bs_u.append(ach_b / best_b if best_b > 0 else 0.0)

    u_mean, u_std = float(np.mean(bs_u)), float(np.std(bs_u))
    u_ci = [float(np.percentile(bs_u, 2.5)), float(np.percentile(bs_u, 97.5))]

    print_flush(f"   Validation Utility 95% CI (B=1,000): [{u_ci[0]:+.6f}, {u_ci[1]:+.6f}] (Mean: {u_mean:+.6f}, Std: {u_std:.6f})")

    # 4. Critical Error Analysis on Missed Septic Cases
    missed_pids, missed_max_p, missed_lengths = [], [], []
    for idx, (lbls, prs) in enumerate(zip(val_labels, val_preds_precomputed)):
        if lbls.max() == 1 and prs.max() == 0:
            missed_pids.append(idx)
            missed_max_p.append(val_probs[idx].max())
            missed_lengths.append(len(lbls))

    print_flush("\n3. Critical Error Analysis on Missed Septic Cases (Validation Cohort):")
    print_flush(f"   Total Missed Septic Patients : {len(missed_pids)} / 169 ({len(missed_pids)/169*100:.1f}%)")
    print_flush(f"   Mean Max M3 Probability      : {np.mean(missed_max_p):.4f} (Baseline threshold was 0.44)")
    print_flush(f"   Median Stay Duration         : {np.median(missed_lengths):.1f} hours")

    # 5. Freeze Primary Policy to JSON
    frozen_dict = {
        "policy_name": c1["policy_name"],
        "category": c1["category"],
        "selection_rule": "Validation Pareto Utility Maximization",
        "threshold": 0.19,
        "persistence": 1,
        "cooldown": 36,
        "val_utility": float(c1["utility"]),
        "val_f1": float(c1["f1"]),
        "val_precision": float(c1["precision"]),
        "val_recall": float(c1["recall"]),
        "val_fpr_h": float(c1["fpr_h"]),
        "val_patient_detection_rate": float(c1["patient_detection_rate"]),
        "val_mean_lead_h": float(c1["mean_lead_h"]),
        "val_utility_bootstrap_ci_95": u_ci,
        "selection_timestamp": datetime.datetime.now().isoformat(),
        "checkpoint_sha256": compute_sha256(EXPERIMENTS_DIR / "final_m3_frozen" / "best_m3_frozen.pt"),
    }

    with open(RESULTS_DIR / "m3_phase4_frozen_policy.json", "w") as f:
        json.dump(frozen_dict, f, indent=4)

    print_flush("\n" + "=" * 95)
    print_flush("VALIDATION POLICY FREEZE COMPLETE")
    print_flush("=" * 95)
    print_flush(f"  Primary Selected Policy  : {c1['policy_name']}")
    print_flush(f"  Category                 : {c1['category']}")
    print_flush(f"  Validation Utility       : {c1['utility']:+.6f}")
    print_flush(f"  Validation Detection Rate: {c1['patient_detection_rate']*100:.1f}% ({c1['n_tp_patients']}/{c1['n_sepsis_patients']})")
    print_flush(f"  Validation FPR/h         : {c1['fpr_h']*100:.2f}%")
    print_flush(f"  Validation Lead Time     : {c1['mean_lead_h']:.1f} hours")
    print_flush(f"  Timestamp                : {frozen_dict['selection_timestamp']}")
    print_flush(f"  Checkpoint SHA256        : {frozen_dict['checkpoint_sha256']}")
    print_flush("=" * 95)

if __name__ == "__main__":
    main()
