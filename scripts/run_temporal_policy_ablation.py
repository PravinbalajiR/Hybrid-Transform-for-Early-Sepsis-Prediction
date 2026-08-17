"""
run_temporal_policy_ablation.py
-------------------------------
Optimized Phase 1 Validation Temporal Policy Ablation & Optimization Script.
Evaluates all candidate temporal alert policies strictly on VALIDATION cohort (N=2,034 patients, 78,755 hours).
Zero test leakage.

Evaluates:
1. Baseline Naive Thresholding
2. Policy A: Persistence (K = 1..6)
3. Policy B: Hysteresis (th_on = 0.30..0.85, th_off = 0.10..0.60)
4. Policy C: Cooldown (C = 2..24h)
5. Policy D: Temporal Smoothing SMA (W = 1..12)
6. Policy E: Exponential Memory EMA (alpha = 0.10..0.90)
7. Policy G: Combined M3-TAP Controller
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

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

RESULTS_DIR = BASE_DIR / "results"

def evaluate_policy_fast(policy, val_labels, val_probs):
    val_preds = policy.generate_alerts_cohort(val_probs)
    u_norm = compute_utility_score(val_labels, val_preds)

    # Flatten for hourly metrics
    y_true_flat = np.concatenate(val_labels)
    y_pred_flat = np.concatenate(val_preds)

    tp_h = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
    fp_h = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
    fn_h = np.sum((y_true_flat == 1) & (y_pred_flat == 0))
    tn_h = np.sum((y_true_flat == 0) & (y_pred_flat == 0))

    prec = tp_h / (tp_h + fp_h) if (tp_h + fp_h) > 0 else 0.0
    rec = tp_h / (tp_h + fn_h) if (tp_h + fn_h) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp_h / (fp_h + tn_h) if (fp_h + tn_h) > 0 else 0.0

    # Patient detection rate
    n_sepsis = 0
    n_tp_sepsis = 0
    for lbls, prs in zip(val_labels, val_preds):
        if lbls.max() == 1:
            n_sepsis += 1
            if prs.max() == 1:
                n_tp_sepsis += 1
    patient_detection_rate = n_tp_sepsis / n_sepsis if n_sepsis > 0 else 0.0

    return {
        "policy_name": policy.name,
        "val_utility": float(u_norm),
        "val_f1": float(f1),
        "val_precision": float(prec),
        "val_recall": float(rec),
        "val_fpr_h": float(fpr),
        "patient_detection_rate": float(patient_detection_rate),
        "n_tp_patients": n_tp_sepsis,
        "n_sepsis_patients": n_sepsis,
    }

def main():
    print("=" * 90)
    print("   OPTIMIZED PHASE 1 VALIDATION TEMPORAL POLICY SWEEP (N=2,034)")
    print("=" * 90)

    val_npz_path = RESULTS_DIR / "m3_final_val_predictions.npz"
    if not val_npz_path.exists():
        print(f"Error: {val_npz_path} not found!")
        sys.exit(1)

    val_data = np.load(val_npz_path, allow_pickle=True)
    val_y_true_flat = val_data["y_true_flat"]
    val_y_proba_flat = val_data["y_proba_flat"]
    val_lengths = val_data["patient_lengths"]

    val_labels = []
    val_probs = []
    curr = 0
    for length in val_lengths:
        val_labels.append(val_y_true_flat[curr : curr + length])
        val_probs.append(val_y_proba_flat[curr : curr + length])
        curr += length

    print(f"Loaded {len(val_labels):,} validation sequences ({len(val_y_true_flat):,} hourly records).\n")

    policies_to_test = []

    # 1. Baseline Naive Sweep
    for th in np.arange(0.10, 0.95, 0.05):
        policies_to_test.append(NaiveThresholdPolicy(threshold=float(th)))

    # 2. Persistence Policy Sweep
    for th in [0.40, 0.44, 0.50, 0.60, 0.70, 0.78, 0.82, 0.85]:
        for K in [1, 2, 3, 4, 5, 6]:
            policies_to_test.append(PersistencePolicy(threshold=float(th), K=int(K)))

    # 3. Hysteresis Policy Sweep
    for th_on in [0.50, 0.60, 0.70, 0.78, 0.82, 0.85, 0.90]:
        for th_off in [0.20, 0.35, 0.50, 0.65]:
            if th_off < th_on:
                policies_to_test.append(HysteresisPolicy(th_on=float(th_on), th_off=float(th_off)))

    # 4. Cooldown Policy Sweep
    for th in [0.44, 0.60, 0.78, 0.82, 0.85]:
        for C in [2, 4, 6, 8, 12, 24]:
            policies_to_test.append(CooldownPolicy(threshold=float(th), cooldown_hours=int(C)))

    # 5. Temporal Smoothing (SMA) Sweep
    for th in [0.40, 0.50, 0.60, 0.70, 0.78, 0.82]:
        for W in [2, 3, 4, 6]:
            policies_to_test.append(MovingAveragePolicy(threshold=float(th), window_K=int(W)))

    # 6. Exponential Memory (EMA) Sweep
    for th in [0.40, 0.50, 0.60, 0.70, 0.78, 0.82]:
        for alpha in [0.20, 0.40, 0.60, 0.80]:
            policies_to_test.append(ExponentialMovingAveragePolicy(threshold=float(th), alpha=float(alpha)))

    # 7. Combined M3-TAP Controller Targeted Grid
    for th_on in [0.60, 0.70, 0.78, 0.82, 0.85, 0.88, 0.90]:
        for th_off in [0.20, 0.35, 0.50]:
            for K in [1, 2, 3]:
                for C in [0, 4, 6, 12]:
                    for W in [1, 2, 3]:
                        policies_to_test.append(
                            CombinedTAPPolicy(
                                th_on=float(th_on),
                                th_off=float(th_off),
                                K_persist=int(K),
                                cooldown_hours=int(C),
                                sma_window=int(W),
                                ema_alpha=0.50,
                            )
                        )

    print(f"Evaluating {len(policies_to_test):,} candidate policies on Validation cohort...")
    records = []
    for idx, pol in enumerate(policies_to_test):
        rec = evaluate_policy_fast(pol, val_labels, val_probs)
        records.append(rec)
        if (idx + 1) % 100 == 0 or (idx + 1) == len(policies_to_test):
            print(f"   Evaluated {idx + 1} / {len(policies_to_test)} policies... Best Val Utility so far: {max(r['val_utility'] for r in records):+.4f}")

    df_results = pd.DataFrame(records)
    out_csv = RESULTS_DIR / "m3_validation_temporal_policy_sweep.csv"
    df_results.to_csv(out_csv, index=False)

    print(f"\nSaved validation policy sweep results to {out_csv}")

    # Top 15 Policies by Validation Utility
    df_top_u = df_results.sort_values(by="val_utility", ascending=False).head(15)
    print("\n[TOP 15 POLICIES BY VALIDATION UTILITY]")
    print(df_top_u[["policy_name", "val_utility", "val_f1", "val_precision", "val_recall", "val_fpr_h", "patient_detection_rate"]].to_string(index=False))

    print("\n" + "=" * 90)
    print("   PHASE 1 VALIDATION TEMPORAL POLICY SWEEP COMPLETE")
    print("=" * 90)

if __name__ == "__main__":
    main()
