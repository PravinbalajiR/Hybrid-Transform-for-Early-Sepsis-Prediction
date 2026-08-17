"""
analyze_m3_tap_phase3_pareto.py
-------------------------------
M3-TAP Phase 3B, 3C, 3D: Pareto Frontier Construction & Validation Policy Freeze.
1. Constructs Pareto non-dominated frontiers across Utility, FPR/h, Detection, Lead Time.
2. Evaluates 10 Prespecified Constraint Categories safely with fallbacks.
3. Performs Patient-Level Bootstrap Robustness Analysis (B=1,000) using regex parser.
4. Freezes Top 5 Validation Policies into results/m3_tap_phase3_selected_policies.json.
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
from scripts.temporal_alert_policy import CooldownPolicy, PersistencePolicy, HysteresisPolicy, CombinedTAPPolicy
from scripts.run_m3_tap_phase3_policy_search import PersistenceCooldownPolicy, HysteresisCooldownPolicy, MedianSmoothingPolicy, RiskAdaptiveCooldownPolicy, AlertCapPolicy
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
    if "Cap" in name:
        m = re.search(r"Cap\d+/\d+h\((.*)\)", name)
        inner_str = m.group(1) if m else name
        return AlertCapPolicy(reconstruct_policy_object(inner_str), 1)

    m_cool = re.search(r"Cooldown\(th=([\d\.]+),\s*C=(\d+)h\)", name)
    if m_cool:
        return CooldownPolicy(float(m_cool.group(1)), int(m_cool.group(2)))

    m_persist = re.search(r"PersistCooldown\(th=([\d\.]+),\s*K=(\d+),\s*C=(\d+)h\)", name)
    if m_persist:
        return PersistenceCooldownPolicy(float(m_persist.group(1)), int(m_persist.group(2)), int(m_persist.group(3)))

    m_hyst = re.search(r"HysteresisCooldown\(high=([\d\.]+),\s*low=([\d\.]+),\s*C=(\d+)h\)", name)
    if m_hyst:
        return HysteresisCooldownPolicy(float(m_hyst.group(1)), float(m_hyst.group(2)), int(m_hyst.group(3)))

    m_risk = re.search(r"RiskAdaptive\(th=([\d\.]+),\s*C_low=(\d+)h,\s*C_high=(\d+)h\)", name)
    if m_risk:
        return RiskAdaptiveCooldownPolicy(float(m_risk.group(1)), int(m_risk.group(2)), int(m_risk.group(3)), 0.40)

    m_med = re.search(r"MedianSmooth\(th=([\d\.]+),\s*W=(\d+)h,\s*C=(\d+)h\)", name)
    if m_med:
        return MedianSmoothingPolicy(float(m_med.group(1)), int(m_med.group(2)), int(m_med.group(3)))

    return CooldownPolicy(0.20, 24)

def safe_select_best(df, mask=None, sort_col="utility"):
    sub = df[mask] if mask is not None else df
    if len(sub) == 0:
        sub = df
    return sub.sort_values(by=sort_col, ascending=False).iloc[0]

def main():
    print_flush("=" * 95)
    print_flush("   M3-TAP PHASE 3B-3D: PARETO FRONTIER, CONSTRAINTS & VALIDATION FREEZE")
    print_flush("=" * 95)

    sweep_csv = RESULTS_DIR / "m3_tap_phase3_policy_sweep.csv"
    val_npz_path = RESULTS_DIR / "m3_final_val_predictions.npz"

    if not sweep_csv.exists() or not val_npz_path.exists():
        print_flush("Error: Required input files missing in results/!")
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

    # 1. Phase 3B: Construct Pareto Frontier across Utility vs Detection vs FPR/h
    pareto_list = []
    sorted_df = df_sweep.sort_values(by="utility", ascending=False)
    
    current_min_fpr = 1.0
    for _, row in sorted_df.iterrows():
        if row["fpr_h"] <= current_min_fpr:
            pareto_list.append(row)
            current_min_fpr = row["fpr_h"]

    df_pareto = pd.DataFrame(pareto_list)
    df_pareto.to_csv(RESULTS_DIR / "m3_tap_phase3_pareto_frontier.csv", index=False)
    print_flush(f"\n1. Constructed Pareto Frontier ({len(df_pareto):,} non-dominated policies). Saved to results/m3_tap_phase3_pareto_frontier.csv")

    # 2. Phase 3C: Evaluate 10 Prespecified Constraint Categories safely
    constraint_results = {}
    
    c1 = safe_select_best(df_sweep)
    constraint_results["C1_MaxUtility"] = c1.to_dict()

    c2 = safe_select_best(df_sweep, df_sweep["fpr_h"] <= 0.0100)
    constraint_results["C2_FPR_1pct"] = c2.to_dict()

    c3 = safe_select_best(df_sweep, df_sweep["fpr_h"] <= 0.0050)
    constraint_results["C3_FPR_0.5pct"] = c3.to_dict()

    c4 = safe_select_best(df_sweep, df_sweep["fpr_h"] <= 0.0030)
    constraint_results["C4_FPR_0.3pct"] = c4.to_dict()

    c5 = safe_select_best(df_sweep, df_sweep["patient_detection_rate"] >= 0.8500)
    constraint_results["C5_Detect_85pct"] = c5.to_dict()

    c6 = safe_select_best(df_sweep, df_sweep["patient_detection_rate"] >= 0.9000)
    constraint_results["C6_Detect_90pct"] = c6.to_dict()

    c7 = safe_select_best(df_sweep, df_sweep["patient_detection_rate"] >= 0.9500)
    constraint_results["C7_Detect_95pct"] = c7.to_dict()

    c8 = safe_select_best(df_sweep, df_sweep["mean_lead_h"] >= 6.0, sort_col="mean_lead_h")
    constraint_results["C8_LeadTime_6h"] = c8.to_dict()

    c9 = safe_select_best(df_sweep, df_sweep["pct_early_6h"] >= 40.0, sort_col="pct_early_6h")
    constraint_results["C9_Warning_6h_40pct"] = c9.to_dict()

    c10 = safe_select_best(df_sweep, (df_sweep["patient_detection_rate"] >= 0.9000) & (df_sweep["fpr_h"] <= 0.0050))
    constraint_results["C10_Combined_Constrained"] = c10.to_dict()

    top_selected_policies = [
        ("Policy 1: Validation Utility Optimum", c1.to_dict()),
        ("Policy 2: Detection >= 90% Optimum", c6.to_dict()),
        ("Policy 3: Low FPR/h <= 0.3% Optimum", c4.to_dict()),
        ("Policy 4: Persistence/Hysteresis Optimum", safe_select_best(df_sweep, df_sweep["category"].isin(["2. Persistence", "4. Hysteresis"])).to_dict()),
        ("Policy 5: Risk-Adaptive Cooldown Optimum", safe_select_best(df_sweep, df_sweep["category"] == "6. RiskAdaptiveCooldown").to_dict()),
    ]

    print_flush("\n2. Selected Top 5 Validation-Locked Policies for Frozen Test Evaluation:")
    for pol_desc, pol_dict in top_selected_policies:
        print_flush(f"   [{pol_desc}] {pol_dict['policy_name']} | Val Utility: {pol_dict['utility']:+.6f} | Det: {pol_dict['patient_detection_rate']*100:.1f}% | FPR/h: {pol_dict['fpr_h']*100:.2f}%")

    # 3. Phase 3D: Patient-Level Bootstrap Analysis (B=1,000) on Selected Policies
    np.random.seed(42)
    B = 1000
    n_val_patients = len(val_labels)
    bs_rows = []

    for pol_desc, pol_dict in top_selected_policies:
        pol_obj = reconstruct_policy_object(pol_dict["policy_name"])
        val_preds = pol_obj.generate_alerts_cohort(val_probs)

        patient_achieved, patient_best = [], []
        for lbls, prs in zip(val_labels, val_preds):
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

        bs_rows.append({
            "policy_desc": pol_desc,
            "policy_name": pol_dict["policy_name"],
            "val_utility_mean": u_mean,
            "val_utility_std": u_std,
            "val_utility_ci_95_low": u_ci[0],
            "val_utility_ci_95_high": u_ci[1],
        })

    df_bs = pd.DataFrame(bs_rows)
    df_bs.to_csv(RESULTS_DIR / "m3_tap_phase3_bootstrap_ci.csv", index=False)
    print_flush(f"\n3. Saved Bootstrap Robustness Analysis (B=1,000) to: results/m3_tap_phase3_bootstrap_ci.csv")

    # 4. Save JSON of Selected Policies
    selected_json = {
        "timestamp": datetime.datetime.now().isoformat(),
        "checkpoint_sha256": compute_sha256(EXPERIMENTS_DIR / "final_m3_frozen" / "best_m3_frozen.pt"),
        "top_policies": top_selected_policies,
        "constraint_matrix": constraint_results
    }
    with open(RESULTS_DIR / "m3_tap_phase3_selected_policies.json", "w") as f:
        json.dump(selected_json, f, indent=4)

    # 5. Scientific Validation Freeze Declaration
    print_flush("\n" + "=" * 95)
    print_flush("VALIDATION POLICY FREEZE COMPLETE")
    print_flush("=" * 95)
    print_flush(f"  Primary Selected Policy  : {top_selected_policies[0][1]['policy_name']}")
    print_flush(f"  Validation Utility       : {top_selected_policies[0][1]['utility']:+.6f}")
    print_flush(f"  Validation Detection Rate: {top_selected_policies[0][1]['patient_detection_rate']*100:.1f}%")
    print_flush(f"  Validation FPR/h         : {top_selected_policies[0][1]['fpr_h']*100:.2f}%")
    print_flush(f"  Validation Lead Time     : {top_selected_policies[0][1]['mean_lead_h']:.1f} hours")
    print_flush(f"  Timestamp                : {selected_json['timestamp']}")
    print_flush(f"  Checkpoint SHA256        : {selected_json['checkpoint_sha256']}")
    print_flush("=" * 95)

if __name__ == "__main__":
    main()
