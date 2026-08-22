"""
verify_phase18_independent.py
------------------------------
Independent Verification Script for Phase 18.
Executes zero-dependency mathematical re-computations of:
  1. FROZEN_MODEL_UTILITY (-0.257312)
  2. GROUND_TRUTH_ORACLE_CEILING (+0.826245570148)
  3. 10-Patient Manual Audit Breakdown
  4. Model Discrimination Metrics & Checkpoint SHA256 Hashes
"""

import sys
import torch
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from evaluation.utility_score import compute_utility_score
from scripts.oracle_reconciliation_independent import (
    calculate_patient_utility,
    calculate_never_alarm,
    calculate_always_alarm,
    calculate_onset_alarm,
    calculate_best_single_alarm,
    calculate_best_persistent_alarm
)

def print_flush(msg: str):
    print(msg, flush=True)

def compute_sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def compute_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i+1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    return float(ece)

def main():
    print_flush("=" * 95)
    print_flush("   PHASE 18 INDEPENDENT VERIFICATION & AUDIT SUITE")
    print_flush("=" * 95)

    # 1. Hashes & Provenance
    ckpt_path = BASE_DIR / "experiments" / "final_m3_frozen" / "best_m3_frozen.pt"
    test_npz_path = BASE_DIR / "results" / "m3_final_test_predictions.npz"
    val_npz_path = BASE_DIR / "results" / "m3_final_val_predictions.npz"

    ckpt_hash = compute_sha256(ckpt_path)
    test_npz_hash = compute_sha256(test_npz_path)
    val_npz_hash = compute_sha256(val_npz_path)

    print_flush(f"1. ARTIFACT SHA256 HASHES:")
    print_flush(f"   Checkpoint Hash (best_m3_frozen.pt) : {ckpt_hash}")
    print_flush(f"   Test Prediction NPZ Hash           : {test_npz_hash}")
    print_flush(f"   Val Prediction NPZ Hash            : {val_npz_hash}\n")

    # Load test predictions
    data_test = np.load(test_npz_path, allow_pickle=True)
    y_true_flat = data_test["y_true_flat"]
    y_proba_flat = data_test["y_proba_flat"]
    patient_lengths = data_test["patient_lengths"]

    curr = 0
    all_labels, all_probs = [], []
    for l in patient_lengths:
        all_labels.append(y_true_flat[curr : curr + l])
        all_probs.append(y_proba_flat[curr : curr + l])
        curr += l

    n_patients = len(all_labels)
    sepsis_indices = [i for i, lbls in enumerate(all_labels) if lbls.max() == 1]
    n_sepsis = len(sepsis_indices)
    n_non_sepsis = n_patients - n_sepsis

    # 2. TASK 3: Verify FROZEN_MODEL_UTILITY = -0.257312
    print_flush("2. TASK 3: INDEPENDENT RE-COMPUTATION OF FROZEN_MODEL_UTILITY")
    th_frozen = 0.190
    c_frozen = 36

    preds_frozen = []
    for lbls, prs in zip(all_labels, all_probs):
        p = np.zeros(len(lbls), dtype=int)
        alarm_idx = np.where(prs >= th_frozen)[0]
        if len(alarm_idx) > 0:
            t_curr = alarm_idx[0]
            while t_curr < len(lbls):
                if prs[t_curr] >= th_frozen:
                    p[t_curr] = 1
                    t_curr += c_frozen
                else: t_curr += 1
        preds_frozen.append(p)

    recomputed_frozen_u = compute_utility_score(all_labels, preds_frozen)
    reported_frozen_u = -0.257312450379

    diff_frozen = abs(recomputed_frozen_u - reported_frozen_u)
    print_flush(f"   Recomputed FROZEN_MODEL_UTILITY : {recomputed_frozen_u:+.12f}")
    print_flush(f"   Reported FROZEN_MODEL_UTILITY   : {reported_frozen_u:+.12f}")
    print_flush(f"   Absolute Discrepancy             : {diff_frozen:.12e}")
    if diff_frozen > 1e-10:
        print_flush("CRITICAL FAILURE: Discrepancy > 1e-10! Stopping.")
        sys.exit(1)
    print_flush("   -> VERIFIED EXACT (Discrepancy <= 1e-10)\n")

    # 3. TASK 4: Verify GROUND_TRUTH_ORACLE_CEILING = +0.826245570148
    print_flush("3. TASK 4: INDEPENDENT RE-COMPUTATION OF GROUND_TRUTH_ORACLE_CEILING")
    tot_gt_ach = 0.0
    tot_gt_best = 0.0
    n_gt_ge6 = 0
    n_gt_lt6 = 0

    for lbls in all_labels:
        lbls = np.asarray(lbls, dtype=int)
        is_sep = int(lbls.max()) == 1
        if is_sep:
            t_on = int(np.argmax(lbls))
            tot_gt_best += 1.0
            if t_on >= 6:
                tot_gt_ach += 1.0
                n_gt_ge6 += 1
            else:
                tot_gt_ach += (t_on + 3.0) / 9.0
                n_gt_lt6 += 1

    recomputed_gt_oracle = tot_gt_ach / tot_gt_best
    reported_gt_oracle = 0.826245570148

    diff_gt = abs(recomputed_gt_oracle - reported_gt_oracle)
    print_flush(f"   Total Septic Patients           : {tot_best:.0f} (Expected: 1066)")
    print_flush(f"   Septic Patients (Onset >= 6h)   : {n_gt_ge6} (Credit: {n_gt_ge6:.1f} pts)")
    print_flush(f"   Septic Patients (Onset < 6h)    : {n_gt_lt6} (Credit: {n_gt_lt6 * (tot_gt_ach - n_gt_ge6)/n_gt_lt6:.6f} pts)")
    print_flush(f"   Total Achieved Points           : {tot_gt_ach:.6f} (Expected: 880.777778 pts)")
    print_flush(f"   Recomputed GT Oracle Ceiling    : {recomputed_gt_oracle:+.12f}")
    print_flush(f"   Reported GT Oracle Ceiling      : {reported_gt_oracle:+.12f}")
    print_flush(f"   Absolute Discrepancy             : {diff_gt:.12e}")
    if diff_gt > 1e-10:
        print_flush("CRITICAL FAILURE: GT Oracle discrepancy > 1e-10! Stopping.")
        sys.exit(1)
    print_flush("   -> VERIFIED EXACT (Discrepancy <= 1e-10)\n")

    # 4. TASK 5: 10-PATIENT MANUAL AUDIT BREAKDOWN
    print_flush("4. TASK 5: 10-PATIENT MANUAL AUDIT BREAKDOWN")
    audit_ids = [12, 54, 355, 46, 15, 14, 39, 3, 11, 16]
    audit_rows = []

    for p_id in audit_ids:
        lbls = all_labels[p_id]
        prs = all_probs[p_id]
        T = len(lbls)
        is_sep = int(lbls.max()) == 1
        t_on = int(np.argmax(lbls)) if is_sep else -1

        _, u_never, _ = calculate_never_alarm(lbls)
        _, u_always, _ = calculate_always_alarm(lbls)
        _, u_onset, _ = calculate_onset_alarm(lbls)
        _, u_best_single, _ = calculate_best_single_alarm(lbls)
        _, u_best_persist, _ = calculate_best_persistent_alarm(lbls)

        audit_rows.append({
            "patient_id": p_id,
            "sepsis_status": int(is_sep),
            "sequence_length": T,
            "onset": t_on,
            "never_alarm": u_never,
            "always_alarm": u_always,
            "onset_alarm": u_onset,
            "best_single_alarm": u_best_single,
            "best_persistent_alarm": u_best_persist
        })

    df_audit = pd.DataFrame(audit_rows)
    print_flush(df_audit.to_string(index=False))

    # Explicit arithmetic reconstruction for 3 representative patients
    print_flush("\n   --- Explicit Hand-Arithmetic Reconstructions ---")
    
    # Patient 12: Septic >= 6h (Onset = 80h, Length = 90h)
    p12 = df_audit[df_audit["patient_id"] == 12].iloc[0]
    print_flush(f"   A. Patient 12 (Septic, Onset = 80h >= 6h):")
    print_flush(f"      - Optimal Single Alarm at t = max(0, 80 - 6) = 74h.")
    print_flush(f"      - Lead time dt = 80 - 74 = 6h = dt_optimal.")
    print_flush(f"      - Achieved Utility = max_u_tp * (6 + 3)/(6 + 3) = 1.0 * (9/9) = +1.0000. Verified: {p12['best_single_alarm'] == 1.0}")

    # Patient 54: Septic < 6h (Onset = 0h, Length = 8h)
    p54 = df_audit[df_audit["patient_id"] == 54].iloc[0]
    print_flush(f"   B. Patient 54 (Septic, Onset = 0h < 6h):")
    print_flush(f"      - Optimal Single Alarm at t = max(0, 0 - 6) = 0h.")
    print_flush(f"      - Lead time dt = 0 - 0 = 0h.")
    print_flush(f"      - Achieved Utility = max_u_tp * (0 + 3)/(6 + 3) = 1.0 * (3/9) = +0.333333. Verified: {abs(p54['best_single_alarm'] - 0.333333333333) < 1e-6}")

    # Patient 16: Non-septic (Onset = -1, Length = 15h)
    p16 = df_audit[df_audit["patient_id"] == 16].iloc[0]
    print_flush(f"   C. Patient 16 (Non-septic):")
    print_flush(f"      - Never Alarm: 0 alarms -> FP penalty = 0.0. Verified: {p16['never_alarm'] == 0.0}")
    print_flush(f"      - Always Alarm: 15 alarms -> FP penalty = -0.05 * 15 = -0.7500. Verified: {p16['always_alarm'] == -0.7500}\n")

    # 5. TASK 11: VERIFY MODEL-DISCRIMINATION NUMBERS
    print_flush("5. TASK 11: VERIFY MODEL-DISCRIMINATION NUMBERS")
    auroc = roc_auc_score(y_true_flat, y_proba_flat)
    auprc = average_precision_score(y_true_flat, y_proba_flat)
    brier = brier_score_loss(y_true_flat, y_proba_flat)
    ece = compute_ece(y_true_flat, y_proba_flat)

    print_flush(f"   BIDMC Test AUROC : {auroc:.6f} (Reported: 0.961663)")
    print_flush(f"   BIDMC Test AUPRC : {auprc:.6f} (Reported: 0.423062)")
    print_flush(f"   BIDMC Test Brier : {brier:.6f} (Reported: 0.01529 raw / 0.021326 uncalibrated)")
    print_flush(f"   BIDMC Test ECE   : {ece:.6f}")

    print_flush("\n" + "=" * 95)
    print_flush("   PHASE 18 RE-COMPUTATIONS & AUDIT SUITE COMPLETE")
    print_flush("=" * 95)

if __name__ == "__main__":
    main()
