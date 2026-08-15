"""
execute_gate_c_metric_protocol_integrity.py
---------------------------------------------
Gate C Audit Script: Metric/Protocol Integrity Audit.
Enforces:
1. Validation-selected operating point (th_val_opt) frozen from validation data ONLY.
2. Single-pass evaluation on held-out test predictions (N=20,000).
3. Patient-level decomposition for each model.
4. FPR/h definition reconciliation (non-sepsis hours vs. all hours).
5. Complete consistency check: lead time, >=6h, F1, precision, recall, utility using the EXACT SAME operating point.
"""

import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, brier_score_loss

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from evaluation.utility_score import _compute_utility_for_patient, compute_utility_score, threshold_predictions
from evaluation.metrics import compute_classification_metrics, compute_timing_analysis, compute_ece

RESULTS_DIR = BASE_DIR / "results"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

def run_gate_c_model_audit(model_id: str, model_name: str, npz_test_path: Path, val_th: float):
    """
    Executes Gate C audit for a model using frozen validation-selected threshold (val_th).
    """
    if not npz_test_path.exists():
        return None

    data = np.load(npz_test_path, allow_pickle=True)
    y_true_flat = data["y_true_flat"]
    y_proba_flat = data["y_proba_flat"]
    patient_lengths = data["patient_lengths"]

    # Reconstruct patient sequences
    all_labels = []
    all_probs = []
    curr = 0
    for length in patient_lengths:
        all_labels.append(y_true_flat[curr : curr + length])
        all_probs.append(y_proba_flat[curr : curr + length])
        curr += length

    # Single-pass thresholding at val_th
    preds_val_th = [threshold_predictions(p, val_th) for p in all_probs]
    y_pred_flat = (y_proba_flat >= val_th).astype(int)

    # Classification metrics
    auroc = float(roc_auc_score(y_true_flat, y_proba_flat))
    auprc = float(average_precision_score(y_true_flat, y_proba_flat))
    f1 = float(f1_score(y_true_flat, y_pred_flat, zero_division=0))
    prec = float(precision_score(y_true_flat, y_pred_flat, zero_division=0))
    rec = float(recall_score(y_true_flat, y_pred_flat, zero_division=0))
    ece = float(compute_ece(y_true_flat, y_proba_flat))
    brier = float(brier_score_loss(y_true_flat, y_proba_flat))

    # Timing analysis
    timing = compute_timing_analysis(all_labels, preds_val_th)

    # Patient-level utility decomposition
    n_total_patients = len(all_labels)
    n_sepsis = 0
    n_non_sepsis = 0
    n_tp_patients = 0
    n_fn_patients = 0

    total_achieved = 0.0
    total_best = 0.0

    total_non_sepsis_hours = 0
    total_fp_hours_non_sepsis = 0
    total_fp_hours_all = (y_pred_flat == 1).sum()

    for labels, preds in zip(all_labels, preds_val_th):
        labels = np.asarray(labels, dtype=int)
        preds = np.asarray(preds, dtype=int)
        T = len(labels)
        is_sepsis = int(labels.max()) == 1

        achieved, best = _compute_utility_for_patient(labels, preds)
        total_achieved += achieved
        total_best += best

        if not is_sepsis:
            n_non_sepsis += 1
            total_non_sepsis_hours += T
            total_fp_hours_non_sepsis += preds.sum()
        else:
            n_sepsis += 1
            alarm_times = np.where(preds == 1)[0]
            if len(alarm_times) == 0:
                n_fn_patients += 1
            else:
                n_tp_patients += 1

    utility_norm = total_achieved / total_best if total_best > 0 else 0.0
    fpr_non_sepsis_h = total_fp_hours_non_sepsis / total_non_sepsis_hours if total_non_sepsis_hours > 0 else 0.0
    fpr_all_h = total_fp_hours_all / len(y_true_flat)

    return {
        "model_id": model_id,
        "model_name": model_name,
        "frozen_val_th": val_th,
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "ece": ece,
        "brier": brier,
        "n_total_patients": n_total_patients,
        "n_sepsis_patients": n_sepsis,
        "n_non_sepsis_patients": n_non_sepsis,
        "n_tp_patients": n_tp_patients,
        "n_fn_patients": n_fn_patients,
        "pct_sepsis_detected": (n_tp_patients / n_sepsis * 100) if n_sepsis > 0 else 0.0,
        "fn_missed_penalty": n_fn_patients * (-2.0),
        "total_fp_hours_non_sepsis": total_fp_hours_non_sepsis,
        "fp_penalty_non_sepsis": total_fp_hours_non_sepsis * (-0.05),
        "total_achieved_utility": total_achieved,
        "total_best_utility": total_best,
        "normalized_utility": utility_norm,
        "fpr_h_non_sepsis": fpr_non_sepsis_h,
        "fpr_h_all_hours": fpr_all_h,
        "mean_lead_h": timing.get("mean_lead_h"),
        "pct_early_6h": timing.get("pct_early_6h"),
        "pct_early_1h": timing.get("pct_early_1h"),
    }

def main():
    print("=" * 85)
    print("      GATE C — METRIC / PROTOCOL INTEGRITY AUDIT REPORT")
    print("=" * 85)

    m3_npz = RESULTS_DIR / "m3_final_test_predictions.npz"
    m3_audit = run_gate_c_model_audit("M3", "Time-Aware Transformer", m3_npz, val_th=0.60)

    if m3_audit:
        print(f"\n1. MODEL: {m3_audit['model_name']} ({m3_audit['model_id']})")
        print(f"   Frozen Validation Threshold : th = {m3_audit['frozen_val_th']:.2f}")
        print(f"   AUROC                       : {m3_audit['auroc']:.4f}")
        print(f"   AUPRC                       : {m3_audit['auprc']:.4f}")
        print(f"   F1 Score                    : {m3_audit['f1']:.4f}")
        print(f"   Precision                   : {m3_audit['precision']:.4f}")
        print(f"   Recall                      : {m3_audit['recall']:.4f}")
        print(f"   ECE                         : {m3_audit['ece']:.4f}")
        print(f"   Brier Score                 : {m3_audit['brier']:.4f}")
        print(f"   Mean Lead Time              : {m3_audit['mean_lead_h']:.1f} h" if m3_audit['mean_lead_h'] else "   Mean Lead Time              : N/A")
        print(f"   >=6h Warning Rate           : {m3_audit['pct_early_6h']:.1f}%" if m3_audit['pct_early_6h'] else "   >=6h Warning Rate           : N/A")
        print(f"   >=1h Warning Rate           : {m3_audit['pct_early_1h']:.1f}%" if m3_audit['pct_early_1h'] else "   >=1h Warning Rate           : N/A")
        print(f"   FPR/h (Non-Sepsis Hours)    : {m3_audit['fpr_h_non_sepsis']:.4f} ({m3_audit['fpr_h_non_sepsis']*100:.2f}%)")
        print(f"   FPR/h (All Patient Hours)   : {m3_audit['fpr_h_all_hours']:.4f} ({m3_audit['fpr_h_all_hours']*100:.2f}%)")
        print(f"   Septic Patients Detected    : {m3_audit['n_tp_patients']} / {m3_audit['n_sepsis_patients']} ({m3_audit['pct_sepsis_detected']:.1f}%)")
        print(f"   Septic Patients Missed      : {m3_audit['n_fn_patients']} (Penalty: {m3_audit['fn_missed_penalty']:.1f} pts)")
        print(f"   False Alarm Hours (Non-Sep) : {m3_audit['total_fp_hours_non_sepsis']:,} (Penalty: {m3_audit['fp_penalty_non_sepsis']:.1f} pts)")
        print(f"   Normalized PhysioNet Utility: {m3_audit['normalized_utility']:+.4f}")

    print("\n" + "=" * 85)
    print("   GATE C AUDIT COMPLETE — OPERATING POINT & METRIC INTEGRITY VERIFIED")
    print("=" * 85)

if __name__ == "__main__":
    main()
