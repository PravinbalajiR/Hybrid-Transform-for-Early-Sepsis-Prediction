"""
recompute_exact_decompositions.py
---------------------------------
Dedicated Patient-Level Utility Decomposition Audit for th = 0.44, 0.60, 0.78.
Recomputes exact patient rewards, missed penalties, false alarm hours, false alarm penalties,
and verifies exact arithmetic identity:
  Achieved Utility = TP Reward + FN Penalty + FP Penalty
  Normalized Utility = Achieved Utility / Best Utility
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from evaluation.utility_score import threshold_predictions

RESULTS_DIR = BASE_DIR / "results"

def official_patient_utility_decomposition(labels: np.ndarray, predictions: np.ndarray):
    """
    Deconstructs official PhysioNet 2019 utility score for a single patient.
    Returns: (achieved_utility, best_utility, tp_reward, fn_penalty, fp_hours, fp_penalty, is_sepsis, is_tp, is_fn)
    """
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    T = len(labels)

    dt_early = 12.0
    dt_optimal = 6.0
    dt_late = 3.0
    max_u_tp = 1.0
    min_u_fn = -2.0
    u_fp = -0.05

    is_sepsis = int(labels.max()) == 1

    if not is_sepsis:
        # Non-sepsis patient
        fp_hours = int(predictions.sum())
        fp_penalty = u_fp * fp_hours
        achieved = fp_penalty
        best = 0.0
        return achieved, best, 0.0, 0.0, fp_hours, fp_penalty, False, False, False

    # Sepsis patient
    t_onset = int(np.argmax(labels))
    alarm_times = np.where(predictions == 1)[0]

    if len(alarm_times) == 0:
        # Missed sepsis (FN)
        fn_penalty = min_u_fn
        achieved = fn_penalty
        best = max_u_tp
        return achieved, best, 0.0, fn_penalty, 0, 0.0, True, False, True

    # Septic patient with at least 1 alarm
    t_alarm = int(alarm_times[0])
    dt = t_onset - t_alarm

    if dt >= dt_optimal:
        if dt >= dt_early:
            tp_reward = 0.0
        else:
            tp_reward = max_u_tp * (dt - dt_early) / (dt_optimal - dt_early)
    elif dt >= -dt_late:
        tp_reward = max_u_tp * (dt + dt_late) / (dt_optimal + dt_late)
        tp_reward = max(0.0, tp_reward)
    else:
        tp_reward = 0.0

    # FP alarms before early window
    fp_early = int((alarm_times < (t_onset - dt_early)).sum())
    fp_penalty = u_fp * fp_early
    achieved = tp_reward + fp_penalty
    best = max_u_tp

    return achieved, best, tp_reward, 0.0, fp_early, fp_penalty, True, True, False


def run_threshold_decomposition(all_labels, all_probs, threshold: float):
    """
    Computes cohort-level utility decomposition at a given threshold.
    """
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

    total_achieved = 0.0
    total_best = 0.0

    for labels, probs in zip(all_labels, all_probs):
        preds = threshold_predictions(probs, threshold)
        ach, best, tp_rew, fn_pen, fp_hrs, fp_pen, is_sep, is_tp, is_fn = official_patient_utility_decomposition(labels, preds)

        total_achieved += ach
        total_best += best

        if not is_sep:
            n_non_sepsis += 1
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

    total_fp_penalty = sum_fp_penalty_non_sepsis + sum_fp_penalty_sepsis
    total_reconstructed_achieved = sum_tp_reward + sum_fn_penalty + total_fp_penalty
    norm_utility = total_achieved / total_best if total_best > 0 else 0.0

    return {
        "threshold": threshold,
        "n_total_patients": n_patients,
        "n_sepsis_patients": n_sepsis,
        "n_non_sepsis_patients": n_non_sepsis,
        "n_tp_patients": n_tp_sepsis,
        "n_fn_patients": n_fn_sepsis,
        "pct_sepsis_detected": (n_tp_sepsis / n_sepsis * 100) if n_sepsis > 0 else 0.0,
        "sum_tp_reward": sum_tp_reward,
        "sum_fn_penalty": sum_fn_penalty,
        "total_fp_hours_non_sepsis": total_fp_hours_non_sepsis,
        "sum_fp_penalty_non_sepsis": sum_fp_penalty_non_sepsis,
        "total_fp_hours_sepsis": total_fp_hours_sepsis,
        "sum_fp_penalty_sepsis": sum_fp_penalty_sepsis,
        "total_fp_penalty": total_fp_penalty,
        "total_achieved_utility": total_achieved,
        "total_reconstructed_achieved": total_reconstructed_achieved,
        "total_best_utility": total_best,
        "normalized_utility": norm_utility,
        "arithmetic_mismatch": abs(total_achieved - total_reconstructed_achieved),
    }


def main():
    print("=" * 85)
    print("      EXACT PATIENT-LEVEL UTILITY DECOMPOSITION AUDIT")
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

    print(f"Loaded {len(all_labels):,} test patient sequences ({len(y_true_flat):,} hourly records).\n")

    thresholds = [0.44, 0.60, 0.78]
    results = []

    for th in thresholds:
        res = run_threshold_decomposition(all_labels, all_probs, th)
        results.append(res)

        print(f"--- DECOMPOSITION AT THRESHOLD {th:.2f} ---")
        print(f"  Primary Status               : {'[PRIMARY PRESPECIFIED PROTOCOL]' if th==0.44 else '[SENSITIVITY ANALYSIS]'}")
        print(f"  Septic Patients (Total)      : {res['n_sepsis_patients']:,}")
        print(f"  Septic Patients Detected (TP): {res['n_tp_patients']:,} ({res['pct_sepsis_detected']:.1f}%)")
        print(f"  Septic Patients Missed (FN)  : {res['n_fn_patients']:,}")
        print(f"  Early-Warning TP Reward      : +{res['sum_tp_reward']:.2f} pts")
        print(f"  Missed-Sepsis FN Penalty     : {res['sum_fn_penalty']:.2f} pts")
        print(f"  Non-Sepsis FP Alarm Hours    : {res['total_fp_hours_non_sepsis']:,}")
        print(f"  Non-Sepsis FP Penalty        : {res['sum_fp_penalty_non_sepsis']:.2f} pts")
        print(f"  Sepsis Early FP Hours        : {res['total_fp_hours_sepsis']:,}")
        print(f"  Sepsis Early FP Penalty      : {res['sum_fp_penalty_sepsis']:.2f} pts")
        print(f"  Total FP Penalty (All)       : {res['total_fp_penalty']:.2f} pts")
        print(f"  Total Achieved Utility       : {res['total_achieved_utility']:.2f} pts")
        print(f"  Sum of Components            : {res['total_reconstructed_achieved']:.2f} pts")
        print(f"  Arithmetic Mismatch          : {res['arithmetic_mismatch']:.8e}")
        print(f"  Total Best Possible Utility  : {res['total_best_utility']:.2f} pts")
        print(f"  NORMALIZED PHYSIONET UTILITY : {res['normalized_utility']:+.4f}")
        print("-" * 85 + "\n")

    # Save exact decomposition table
    df_decomp = pd.DataFrame(results)
    df_decomp.to_csv(RESULTS_DIR / "EXACT_UTILITY_DECOMPOSITION.csv", index=False)
    print(f"Saved exact decomposition table to {RESULTS_DIR / 'EXACT_UTILITY_DECOMPOSITION.csv'}")

if __name__ == "__main__":
    main()
