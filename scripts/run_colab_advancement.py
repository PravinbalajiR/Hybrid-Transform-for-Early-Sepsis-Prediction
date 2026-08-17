"""
run_colab_advancement.py
------------------------
Master Colab Advancement Script for M3 Temporal Alert Policy (M3-TAP).
Runs complete Phase 1 Validation Policy Sweep, freezes optimal policy, and evaluates single-pass
on held-out test cohort (N=20,000) using official PhysioNet scorer.

Can be run locally or in Google Colab:
  python scripts/run_colab_advancement.py
"""

import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from evaluation.utility_score import compute_utility_score
from evaluation.metrics import compute_timing_analysis
from scripts.temporal_alert_policy import (
    NaiveThresholdPolicy,
    PersistencePolicy,
    HysteresisPolicy,
    CooldownPolicy,
    MovingAveragePolicy,
    ExponentialMovingAveragePolicy,
    CombinedTAPPolicy,
)
from scripts.recompute_exact_decompositions import official_patient_utility_decomposition

RESULTS_DIR = BASE_DIR / "results"

def evaluate_policy_on_cohort(policy, all_labels, all_probs):
    all_preds = policy.generate_alerts_cohort(all_probs)
    u_norm = compute_utility_score(all_labels, all_preds)

    y_true_flat = np.concatenate(all_labels)
    y_pred_flat = np.concatenate(all_preds)

    tp_h = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
    fp_h = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
    fn_h = np.sum((y_true_flat == 1) & (y_pred_flat == 0))
    tn_h = np.sum((y_true_flat == 0) & (y_pred_flat == 0))

    prec = tp_h / (tp_h + fp_h) if (tp_h + fp_h) > 0 else 0.0
    rec = tp_h / (tp_h + fn_h) if (tp_h + fn_h) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp_h / (fp_h + tn_h) if (fp_h + tn_h) > 0 else 0.0

    timing = compute_timing_analysis(all_labels, all_preds)

    n_sepsis = 0
    n_tp_sepsis = 0
    for lbls, prs in zip(all_labels, all_preds):
        if lbls.max() == 1:
            n_sepsis += 1
            if prs.max() == 1:
                n_tp_sepsis += 1

    patient_detection_rate = n_tp_sepsis / n_sepsis if n_sepsis > 0 else 0.0

    return {
        "policy": policy,
        "policy_name": policy.name,
        "utility": float(u_norm),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "fpr_h": float(fpr),
        "patient_detection_rate": float(patient_detection_rate),
        "n_tp_patients": n_tp_sepsis,
        "n_sepsis_patients": n_sepsis,
        "mean_lead_h": timing.get("mean_lead_h"),
        "pct_early_6h": timing.get("pct_early_6h"),
        "pct_early_1h": timing.get("pct_early_1h"),
    }

def main():
    print("=" * 90)
    print("   MASTER M3-TAP ADVANCEMENT EXPERIMENT (COLAB & LOCAL PIPELINE)")
    print("=" * 90)

    val_npz_path = RESULTS_DIR / "m3_final_val_predictions.npz"
    test_npz_path = RESULTS_DIR / "m3_final_test_predictions.npz"

    if not val_npz_path.exists() or not test_npz_path.exists():
        print("Error: Required prediction artifacts missing in results/!")
        sys.exit(1)

    # 1. Load Validation Data (N=2,034)
    val_data = np.load(val_npz_path, allow_pickle=True)
    val_y_true = val_data["y_true_flat"]
    val_y_prob = val_data["y_proba_flat"]
    val_lens = val_data["patient_lengths"]

    val_labels, val_probs = [], []
    curr = 0
    for l in val_lens:
        val_labels.append(val_y_true[curr : curr + l])
        val_probs.append(val_y_prob[curr : curr + l])
        curr += l

    print(f"\n1. Loaded Validation Cohort : {len(val_labels):,} patients ({len(val_y_true):,} hourly records).")

    # 2. Phase 1 Validation Policy Sweep
    print("\n2. Executing Phase 1 Validation Temporal Policy Sweep...")
    candidate_policies = []

    # Naive baseline
    for th in np.arange(0.10, 0.95, 0.05):
        candidate_policies.append(NaiveThresholdPolicy(float(th)))

    # Persistence
    for th in [0.40, 0.44, 0.50, 0.60, 0.70, 0.78, 0.82, 0.85]:
        for K in [1, 2, 3, 4, 5, 6]:
            candidate_policies.append(PersistencePolicy(float(th), int(K)))

    # Hysteresis
    for th_on in [0.50, 0.60, 0.70, 0.78, 0.82, 0.85, 0.90]:
        for th_off in [0.20, 0.35, 0.50, 0.65]:
            if th_off < th_on:
                candidate_policies.append(HysteresisPolicy(float(th_on), float(th_off)))

    # Cooldown
    for th in [0.44, 0.60, 0.78, 0.82, 0.85]:
        for C in [2, 4, 6, 8, 12, 24]:
            candidate_policies.append(CooldownPolicy(float(th), int(C)))

    # SMA / EMA
    for th in [0.40, 0.50, 0.60, 0.70, 0.78, 0.82]:
        for W in [2, 3, 4, 6]:
            candidate_policies.append(MovingAveragePolicy(float(th), int(W)))
        for alpha in [0.20, 0.40, 0.60, 0.80]:
            candidate_policies.append(ExponentialMovingAveragePolicy(float(th), float(alpha)))

    # Combined TAP
    for th_on in [0.60, 0.70, 0.78, 0.82, 0.85, 0.88, 0.90]:
        for th_off in [0.20, 0.35, 0.50]:
            for K in [1, 2, 3]:
                for C in [0, 4, 6, 12]:
                    for W in [1, 2, 3]:
                        candidate_policies.append(CombinedTAPPolicy(float(th_on), float(th_off), int(K), int(C), int(W), 0.50))

    val_results = []
    for pol in candidate_policies:
        val_results.append(evaluate_policy_on_cohort(pol, val_labels, val_probs))

    df_val = pd.DataFrame(val_results)
    df_val_sorted = df_val.sort_values(by="utility", ascending=False)
    
    print("\n   [TOP 10 TEMPORAL POLICIES ON VALIDATION DATA]")
    print(df_val_sorted[["policy_name", "utility", "f1", "precision", "recall", "fpr_h", "patient_detection_rate"]].head(10).to_string(index=False))

    # 3. Select Validation Utility-Optimal Policy
    best_val_row = df_val_sorted.iloc[0]
    best_policy = best_val_row["policy"]
    
    print(f"\n3. Validation Utility-Optimal Policy Selected (Zero Test Leakage):")
    print(f"   Selected Policy : {best_policy.name}")
    print(f"   Val Utility     : {best_val_row['utility']:+.4f}")
    print(f"   Val F1          : {best_val_row['f1']:.4f}")
    print(f"   Val FPR/h       : {best_val_row['fpr_h']:.4f}")

    # Save frozen policy JSON
    frozen_json = {
        "policy_name": best_policy.name,
        "val_utility": float(best_val_row["utility"]),
        "val_f1": float(best_val_row["f1"]),
        "val_precision": float(best_val_row["precision"]),
        "val_recall": float(best_val_row["recall"]),
        "val_fpr_h": float(best_val_row["fpr_h"]),
        "val_patient_detection_rate": float(best_val_row["patient_detection_rate"])
    }
    with open(RESULTS_DIR / "m3_tap_frozen_policy.json", "w") as f:
        json.dump(frozen_json, f, indent=4)
    print(f"   Saved frozen policy configuration to: {RESULTS_DIR / 'm3_tap_frozen_policy.json'}")

    # 4. Phase 2: Single-Pass Evaluation on Held-Out Test Cohort (N=20,000)
    test_data = np.load(test_npz_path, allow_pickle=True)
    test_y_true = test_data["y_true_flat"]
    test_y_prob = test_data["y_proba_flat"]
    test_lens = test_data["patient_lengths"]

    test_labels, test_probs = [], []
    curr = 0
    for l in test_lens:
        test_labels.append(test_y_true[curr : curr + l])
        test_probs.append(test_y_prob[curr : curr + l])
        curr += l

    print(f"\n4. Evaluating Frozen Policy Single-Pass on Held-Out Test Cohort (N={len(test_labels):,} patients)...")
    test_metrics = evaluate_policy_on_cohort(best_policy, test_labels, test_probs)

    print("\n" + "=" * 90)
    print("      HELD-OUT TEST COHORT ADVANCEMENT EVALUATION (N=20,000)")
    print("=" * 90)
    print(f"  Frozen Policy Name           : {test_metrics['policy_name']}")
    print(f"  OFFICIAL TEST UTILITY SCORE  : {test_metrics['utility']:+.4f}  (Baseline th=0.44 was -1.1440)")
    print(f"  Test Hourly F1 Score         : {test_metrics['f1']:.4f}")
    print(f"  Test Hourly Precision (PPV)  : {test_metrics['precision']:.4f}")
    print(f"  Test Hourly Recall           : {test_metrics['recall']:.4f}")
    print(f"  Test Non-Sepsis FPR/h        : {test_metrics['fpr_h']:.4f} ({test_metrics['fpr_h']*100:.2f}%)")
    print(f"  Test Patient Detection Rate  : {test_metrics['patient_detection_rate']:.4f} ({test_metrics['patient_detection_rate']*100:.1f}%) [{test_metrics['n_tp_patients']}/1066 detected]")
    print(f"  Mean Early Warning Lead Time : {test_metrics['mean_lead_h']:.1f} hours" if test_metrics['mean_lead_h'] else "  Mean Early Warning Lead Time : N/A")
    print(f"  >=6h Early Warning Rate      : {test_metrics['pct_early_6h']:.1f}%" if test_metrics['pct_early_6h'] else "  >=6h Early Warning Rate      : N/A")

    # 5. Patient-Level Utility Decomposition on Test Cohort
    test_preds = best_policy.generate_alerts_cohort(test_probs)
    n_tp, n_fn = 0, 0
    tp_reward, fn_penalty, fp_penalty = 0.0, 0.0, 0.0
    fp_hours_non_sep = 0
    total_achieved, total_best = 0.0, 0.0

    for lbls, prs in zip(test_labels, test_preds):
        ach, best, tp_rew, fn_pen, fp_hrs, fp_pen, is_sep, is_tp, is_fn = official_patient_utility_decomposition(lbls, prs)
        total_achieved += ach
        total_best += best
        if is_sep:
            tp_reward += tp_rew
            fn_penalty += fn_pen
            if is_tp: n_tp += 1
            if is_fn: n_fn += 1
        else:
            fp_hours_non_sep += fp_hrs
            fp_penalty += fp_pen

    print("\n" + "=" * 90)
    print("      HELD-OUT TEST PATIENT-LEVEL UTILITY DECOMPOSITION")
    print("=" * 90)
    print(f"  Septic Patients Detected (TP): {n_tp:,} / 1,066 ({n_tp/1066*100:.1f}%)")
    print(f"  Septic Patients Missed (FN)  : {n_fn:,} / 1,066 ({n_fn/1066*100:.1f}%)")
    print(f"  Early Warning TP Reward      : +{tp_reward:.2f} pts")
    print(f"  Missed Sepsis FN Penalty     : {fn_penalty:.2f} pts")
    print(f"  Non-Sepsis False Alarm Hours : {fp_hours_non_sep:,} hours (Penalty: {fp_penalty:.2f} pts)")
    print(f"  Total Achieved Utility (Raw) : {total_achieved:.2f} pts")
    print(f"  Total Best Possible Utility  : {total_best:.2f} pts")
    print(f"  NORMALIZED PHYSIONET UTILITY : {total_achieved/total_best:+.4f}")
    print("=" * 90)

    # Save summary CSV
    summary_df = pd.DataFrame([{
        "policy_name": test_metrics['policy_name'],
        "val_utility": float(best_val_row['utility']),
        "test_utility": float(test_metrics['utility']),
        "test_f1": float(test_metrics['f1']),
        "test_precision": float(test_metrics['precision']),
        "test_recall": float(test_metrics['recall']),
        "test_patient_detection_rate": float(test_metrics['patient_detection_rate']),
        "test_fpr_h": float(test_metrics['fpr_h']),
        "mean_lead_h": test_metrics['mean_lead_h'],
        "pct_early_6h": test_metrics['pct_early_6h'],
    }])
    summary_df.to_csv(RESULTS_DIR / "M3_TAP_ADVANCEMENT_RESULTS.csv", index=False)
    print(f"\nSaved advancement results to {RESULTS_DIR / 'M3_TAP_ADVANCEMENT_RESULTS.csv'}")

if __name__ == "__main__":
    main()
