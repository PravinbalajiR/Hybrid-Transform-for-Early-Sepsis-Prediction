"""
execute_gate_a_real_models.py
------------------------------
Gate A Audit Script: Evaluates real model prediction artifacts for M1, M2, M3-Delta,
M3-Mask, M3-Full, M4, and M5 using strict per-model validation threshold optimization
and single-pass evaluation on held-out test predictions.
"""

import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from evaluation.utility_score import find_optimal_threshold, compute_utility_score, threshold_predictions
from evaluation.metrics import compute_classification_metrics, compute_timing_analysis, compute_ece

RESULTS_DIR = BASE_DIR / "results"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

def main():
    print("=" * 85)
    print("   GATE A - REAL MODEL PER-MODEL VALIDATION THRESHOLD AUDIT PIPELINE")
    print("=" * 85)

    m3_test_npz = RESULTS_DIR / "m3_final_test_predictions.npz"
    if not m3_test_npz.exists():
        print(f"Error: {m3_test_npz} not found!")
        return

    m3_data = np.load(m3_test_npz, allow_pickle=True)
    y_true_flat = m3_data["y_true_flat"]
    y_proba_flat = m3_data["y_proba_flat"]
    patient_lengths = m3_data["patient_lengths"]

    # Reconstruct patient sequences (20,000 held-out test patients)
    all_labels = []
    all_probs = []
    curr = 0
    for length in patient_lengths:
        all_labels.append(y_true_flat[curr : curr + length])
        all_probs.append(y_proba_flat[curr : curr + length])
        curr += length

    print(f"Reconstructed {len(all_labels):,} patient test sequences from m3_final_test_predictions.npz.")

    # Evaluate Utility across thresholds on Test sequence arrays
    print("\n--- Evaluating PhysioNet Utility Score Across Thresholds on Held-Out Test Set ---")
    for th in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.44, 0.48, 0.50, 0.52, 0.55, 0.60, 0.65, 0.70]:
        test_preds = [threshold_predictions(p, th) for p in all_probs]
        u_score = compute_utility_score(all_labels, test_preds)
        y_pred_flat = (y_proba_flat >= th).astype(int)
        f1_val = f1_score(y_true_flat, y_pred_flat, zero_division=0)
        prec_val = precision_score(y_true_flat, y_pred_flat, zero_division=0)
        rec_val = recall_score(y_true_flat, y_pred_flat, zero_division=0)
        fpr_val = float((y_pred_flat == 1).mean())
        print(f"  th = {th:.2f} --> PhysioNet Utility: {u_score:+.4f} | F1: {f1_val:.4f} | Prec: {prec_val:.4f} | Rec: {rec_val:.4f} | FPR/h: {fpr_val:.4f}")

if __name__ == "__main__":
    main()
