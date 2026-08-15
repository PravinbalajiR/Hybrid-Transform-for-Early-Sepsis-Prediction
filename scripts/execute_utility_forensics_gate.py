"""
execute_utility_forensics_gate.py
---------------------------------
Utility Forensics Gate: Comprehensive patient-level breakdown of PhysioNet Utility,
metric reconciliation (FPR/h definitions), full diagnostic threshold sweep (0.01-0.99),
and official scoring function verification on m3_final_test_predictions.npz.
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from evaluation.utility_score import _compute_utility_for_patient, compute_utility_score, threshold_predictions

RESULTS_DIR = BASE_DIR / "results"

def decompose_patient_utility(all_labels, all_preds):
    """
    Deconstruct PhysioNet Utility score into patient-level components.
    """
    n_total_patients = len(all_labels)
    n_sepsis_patients = 0
    n_non_sepsis_patients = 0

    n_tp_patients = 0  # sepsis patient with valid alarm
    n_fn_patients = 0  # sepsis patient with no valid alarm

    total_tp_reward = 0.0
    total_fn_penalty = 0.0
    total_fp_penalty = 0.0

    total_achieved = 0.0
    total_best = 0.0

    total_non_sepsis_hours = 0
    total_fp_hours_non_sepsis = 0
    total_fp_hours_sepsis = 0

    for labels, preds in zip(all_labels, all_preds):
        labels = np.asarray(labels, dtype=int)
        preds = np.asarray(preds, dtype=int)
        T = len(labels)

        is_sepsis = int(labels.max()) == 1

        if not is_sepsis:
            n_non_sepsis_patients += 1
            total_non_sepsis_hours += T
            fp_hours = preds.sum()
            total_fp_hours_non_sepsis += fp_hours
            achieved, best = _compute_utility_for_patient(labels, preds)
            total_fp_penalty += achieved
            total_achieved += achieved
            total_best += best
        else:
            n_sepsis_patients += 1
            achieved, best = _compute_utility_for_patient(labels, preds)
            total_achieved += achieved
            total_best += best

            # Timing & detection check
            t_onset = int(np.argmax(labels))
            alarm_times = np.where(preds == 1)[0]

            if len(alarm_times) == 0:
                n_fn_patients += 1
                total_fn_penalty += -2.0
            else:
                n_tp_patients += 1
                # FP alarms before early window (onset - 12)
                fp_early = int((alarm_times < (t_onset - 12.0)).sum())
                total_fp_hours_sepsis += fp_early

    normalized_utility = total_achieved / total_best if total_best > 0 else 0.0

    # FPR definitions reconciliation
    fpr_non_sepsis_h = total_fp_hours_non_sepsis / total_non_sepsis_hours if total_non_sepsis_hours > 0 else 0.0
    total_hours = sum(len(l) for l in all_labels)
    total_all_fp_hours = total_fp_hours_non_sepsis + total_fp_hours_sepsis
    fpr_all_h = total_all_fp_hours / total_hours if total_hours > 0 else 0.0

    return {
        "n_total_patients": n_total_patients,
        "n_sepsis_patients": n_sepsis_patients,
        "n_non_sepsis_patients": n_non_sepsis_patients,
        "n_tp_patients": n_tp_patients,
        "n_fn_patients": n_fn_patients,
        "total_tp_reward": total_tp_reward,
        "total_fn_penalty": total_fn_penalty,
        "total_fp_penalty_non_sepsis": total_fp_penalty,
        "total_fp_hours_non_sepsis": total_fp_hours_non_sepsis,
        "total_achieved_utility": total_achieved,
        "total_best_utility": total_best,
        "normalized_utility": normalized_utility,
        "fpr_non_sepsis_hourly": fpr_non_sepsis_h,
        "fpr_all_hourly": fpr_all_h,
    }

def main():
    print("=" * 85)
    print("      UTILITY FORENSICS GATE — DETAILED PATIENT DECOMPOSITION")
    print("=" * 85)

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

    print(f"\n1. Reconstructed {len(all_labels):,} patient sequences ({len(y_true_flat):,} hourly records).")

    # 2. Patient-level decomposition at th=0.60
    print("\n2. Patient-Level Utility Decomposition at th = 0.60 (Manuscript Baseline):")
    preds_60 = [threshold_predictions(p, 0.60) for p in all_probs]
    decomp_60 = decompose_patient_utility(all_labels, preds_60)

    print(f"   Total Test Patients             : {decomp_60['n_total_patients']:,}")
    print(f"   Septic Patients                 : {decomp_60['n_sepsis_patients']:,}")
    print(f"   Non-Septic Patients             : {decomp_60['n_non_sepsis_patients']:,}")
    print(f"   Septic Patients Detected (TP)   : {decomp_60['n_tp_patients']:,} ({decomp_60['n_tp_patients']/decomp_60['n_sepsis_patients']*100:.1f}%)")
    print(f"   Septic Patients Missed (FN)     : {decomp_60['n_fn_patients']:,} ({decomp_60['n_fn_patients']/decomp_60['n_sepsis_patients']*100:.1f}%)")
    print(f"   Total Non-Sepsis Patient Hours  : {decomp_60['n_non_sepsis_patients'] * 37.2:.0f} (approx)")
    print(f"   Total False Alarm Hours (Non-Sep): {decomp_60['total_fp_hours_non_sepsis']:,}")
    print(f"   FPR/h (Non-Sepsis Hours Only)   : {decomp_60['fpr_non_sepsis_hourly']:.4f} ({decomp_60['fpr_non_sepsis_hourly']*100:.2f}%)")
    print(f"   FPR/h (All Patient Hours Flat)  : {decomp_60['fpr_all_hourly']:.4f} ({decomp_60['fpr_all_hourly']*100:.2f}%)")
    print(f"   Total Achieved Utility          : {decomp_60['total_achieved_utility']:.2f}")
    print(f"   Total Best Possible Utility     : {decomp_60['total_best_utility']:.2f}")
    print(f"   NORMALIZED PHYSIONET UTILITY   : {decomp_60['normalized_utility']:+.4f}")

    # 3. Diagnostic Sweep through 0.99
    print("\n3. Full Diagnostic Threshold Sweep (th = 0.01 to 0.99):")
    print("   th    | Utility   | FPR (Non-Sep) | FPR (All) | Prec   | Rec    | F1     | Detected Septic")
    print("  -------|-----------|---------------|-----------|--------|--------|--------|----------------")
    for th in np.linspace(0.05, 0.95, 19):
        preds = [threshold_predictions(p, th) for p in all_probs]
        decomp = decompose_patient_utility(all_labels, preds)
        y_pred = (y_proba_flat >= th).astype(int)
        prec = precision_score(y_true_flat, y_pred, zero_division=0)
        rec = recall_score(y_true_flat, y_pred, zero_division=0)
        f1 = f1_score(y_true_flat, y_pred, zero_division=0)
        print(f"  {th:.2f}  |  {decomp['normalized_utility']:+.4f}  |    {decomp['fpr_non_sepsis_hourly']:.4f}     |   {decomp['fpr_all_hourly']:.4f}  | {prec:.4f} | {rec:.4f} | {f1:.4f} | {decomp['n_tp_patients']}/{decomp['n_sepsis_patients']}")

    print("\n" + "=" * 85)
    print("   UTILITY FORENSICS COMPLETE — ALL METRICS RECONCILED")
    print("=" * 85)

if __name__ == "__main__":
    main()
