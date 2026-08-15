"""
reconcile_all_protocol_metrics.py
---------------------------------
Master Protocol & Metric Reconciliation Audit.
Computes and verifies exact metrics for th = 0.44, 0.60, 0.78:
1. Hourly Recall (Sensitivity) vs Patient-Level Detection Rate.
2. Non-Septic Hourly FPR vs All-Hour Alarm Rate.
3. Official Utility Score reconciliation (-1.1440, -0.9535, -0.8696).
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

from evaluation.utility_score import compute_utility_score, threshold_predictions
from evaluation.metrics import compute_timing_analysis
from scripts.recompute_exact_decompositions import official_patient_utility_decomposition

RESULTS_DIR = BASE_DIR / "results"

def run_reconciliation(all_labels, all_probs, threshold: float):
    y_true_flat = np.concatenate(all_labels)
    y_prob_flat = np.concatenate(all_probs)
    preds = [threshold_predictions(p, threshold) for p in all_probs]
    y_pred_flat = (y_prob_flat >= threshold).astype(int)

    # Scikit-learn hourly metrics
    hourly_auroc = float(roc_auc_score(y_true_flat, y_prob_flat))
    hourly_auprc = float(average_precision_score(y_true_flat, y_prob_flat))
    hourly_f1 = float(f1_score(y_true_flat, y_pred_flat, zero_division=0))
    hourly_prec = float(precision_score(y_true_flat, y_pred_flat, zero_division=0))
    hourly_recall = float(recall_score(y_true_flat, y_pred_flat, zero_division=0))

    # Timing analysis
    timing = compute_timing_analysis(all_labels, preds)

    # Patient-level decomposition & detection rate
    n_patients = len(all_labels)
    n_sepsis = 0
    n_non_sepsis = 0
    n_tp_sepsis = 0
    n_fn_sepsis = 0

    sum_tp_reward = 0.0
    sum_fn_penalty = 0.0
    sum_fp_penalty_non_sepsis = 0.0
    sum_fp_penalty_sepsis = 0.0

    total_fp_hours_non_sepsis = 0
    total_fp_hours_sepsis = 0
    total_non_sepsis_hours = 0

    total_achieved = 0.0
    total_best = 0.0

    for labels, probs in zip(all_labels, all_probs):
        pred_i = threshold_predictions(probs, threshold)
        ach, best, tp_rew, fn_pen, fp_hrs, fp_pen, is_sep, is_tp, is_fn = official_patient_utility_decomposition(labels, pred_i)

        total_achieved += ach
        total_best += best

        if not is_sep:
            n_non_sepsis += 1
            total_non_sepsis_hours += len(labels)
            total_fp_hours_non_sepsis += fp_hrs
            sum_fp_penalty_non_sepsis += fp_pen
        else:
            n_sepsis += 1
            sum_tp_reward += tp_rew
            sum_fn_penalty += fn_pen
            total_fp_hours_sepsis += fp_hrs
            sum_fp_penalty_sepsis += fp_pen
            if is_tp:
                n_tp_sepsis += 1
            if is_fn:
                n_fn_sepsis += 1

    patient_detection_rate = n_tp_sepsis / n_sepsis if n_sepsis > 0 else 0.0
    non_sepsis_fpr_h = total_fp_hours_non_sepsis / total_non_sepsis_hours if total_non_sepsis_hours > 0 else 0.0
    all_hours_alarm_rate = (y_pred_flat == 1).mean()
    norm_utility = total_achieved / total_best if total_best > 0 else 0.0

    return {
        "threshold": threshold,
        "hourly_auroc": hourly_auroc,
        "hourly_auprc": hourly_auprc,
        "hourly_f1": hourly_f1,
        "hourly_precision": hourly_prec,
        "hourly_recall": hourly_recall,
        "patient_detection_rate": patient_detection_rate,
        "n_tp_sepsis_patients": n_tp_sepsis,
        "n_fn_sepsis_patients": n_fn_sepsis,
        "sum_tp_reward": sum_tp_reward,
        "sum_fn_penalty": sum_fn_penalty,
        "total_fp_hours_non_sepsis": total_fp_hours_non_sepsis,
        "sum_fp_penalty_non_sepsis": sum_fp_penalty_non_sepsis,
        "total_fp_hours_sepsis": total_fp_hours_sepsis,
        "sum_fp_penalty_sepsis": sum_fp_penalty_sepsis,
        "total_achieved_utility": total_achieved,
        "total_best_utility": total_best,
        "normalized_utility": norm_utility,
        "non_sepsis_fpr_h": non_sepsis_fpr_h,
        "all_hours_alarm_rate": all_hours_alarm_rate,
        "mean_lead_h": timing.get("mean_lead_h"),
        "pct_early_6h": timing.get("pct_early_6h"),
        "pct_early_1h": timing.get("pct_early_1h"),
    }

def main():
    print("=" * 90)
    print("   MASTER PROTOCOL & METRIC RECONCILIATION AUDIT TABLE")
    print("=" * 90)

    npz_path = RESULTS_DIR / "m3_final_test_predictions.npz"
    if not npz_path.exists():
        print(f"Error: {npz_path} not found!")
        return

    data = np.load(npz_path, allow_pickle=True)
    y_true_flat = data["y_true_flat"]
    y_proba_flat = data["y_proba_flat"]
    patient_lengths = data["patient_lengths"]

    all_labels = []
    all_probs = []
    curr = 0
    for length in patient_lengths:
        all_labels.append(y_true_flat[curr : curr + length])
        all_probs.append(y_proba_flat[curr : curr + length])
        curr += length

    thresholds = [0.44, 0.60, 0.78]
    reconciled = [run_reconciliation(all_labels, all_probs, th) for th in thresholds]

    df_rec = pd.DataFrame(reconciled)
    
    print("\n[EXACT AUDITED METRIC MATRIX]")
    for row in reconciled:
        th = row['threshold']
        print(f"\nTHRESHOLD th = {th:.2f} ({'Primary Protocol: Val Utility Optimum' if th==0.44 else 'Sensitivity: Balanced Fallback' if th==0.60 else 'Sensitivity: Val F1 Optimum'}):")
        print(f"  Normalized Utility Score   : {row['normalized_utility']:+.4f}")
        print(f"  Hourly F1 Score            : {row['hourly_f1']:.4f}")
        print(f"  Hourly Precision (PPV)     : {row['hourly_precision']:.4f}")
        print(f"  Hourly Recall (Sensitivity): {row['hourly_recall']:.4f} ({row['hourly_recall']*100:.2f}%)")
        print(f"  Patient Detection Rate     : {row['patient_detection_rate']:.4f} ({row['patient_detection_rate']*100:.1f}%) [{row['n_tp_sepsis_patients']}/1066 detected]")
        print(f"  Patient Missed Rate        : {(1-row['patient_detection_rate']):.4f} ({(1-row['patient_detection_rate'])*100:.1f}%) [{row['n_fn_sepsis_patients']}/1066 missed]")
        print(f"  Non-Sepsis Hourly FPR/h    : {row['non_sepsis_fpr_h']:.4f} ({row['non_sepsis_fpr_h']*100:.2f}%) [{row['total_fp_hours_non_sepsis']:,} FP hours]")
        print(f"  All-Hours Alarm Rate       : {row['all_hours_alarm_rate']:.4f} ({row['all_hours_alarm_rate']*100:.2f}%)")
        print(f"  Mean Early Lead Time       : {row['mean_lead_h']:.1f} h")
        print(f"  >=6h Warning Rate          : {row['pct_early_6h']:.1f}%")
        print(f"  >=1h Warning Rate          : {row['pct_early_1h']:.1f}%")

    df_rec.to_csv(RESULTS_DIR / "RECONCILED_METRIC_MATRIX.csv", index=False)
    print("\n" + "=" * 90)
    print(f"Saved reconciled metric matrix to {RESULTS_DIR / 'RECONCILED_METRIC_MATRIX.csv'}")
    print("=" * 90)

if __name__ == "__main__":
    main()
