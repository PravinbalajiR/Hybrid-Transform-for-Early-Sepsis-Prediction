"""
test_utility_postprocessing.py
------------------------------
Diagnostic script testing moving average smoothing and hysteresis post-processing
on validation predictions to evaluate PhysioNet Utility optimization.
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from evaluation.utility_score import compute_utility_score, threshold_predictions

DATA_DIR = BASE_DIR / "data" / "processed"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

def apply_moving_average_postprocessing(probs_list, window=3):
    """Apply moving average smoothing over sequence window."""
    smoothed = []
    for p in probs_list:
        if len(p) < window:
            sm = p
        else:
            # Moving average
            weights = np.ones(window) / window
            sm = np.convolve(p, weights, mode='same')
        smoothed.append(sm)
    return smoothed

def apply_single_alarm_postprocessing(preds_list):
    """Keep only the first alarm per patient sequence."""
    single_preds = []
    for pred in preds_list:
        p_out = np.zeros_like(pred)
        alarm_idx = np.where(pred == 1)[0]
        if len(alarm_idx) > 0:
            p_out[alarm_idx[0]] = 1
        single_preds.append(p_out)
    return single_preds

def main():
    print("=" * 75)
    print("   DIAGNOSTIC: POST-PROCESSING & UTILITY SCORE OPTIMIZATION")
    print("=" * 75)

    val_sweep_path = BASE_DIR / "results" / "m3_validation_threshold_sweep.csv"
    if val_sweep_path.exists():
        df_val = pd.read_csv(val_sweep_path)
        best_row = df_val.loc[df_val['utility'].idxmax()]
        print("\nRaw Validation Sweep (m3_validation_threshold_sweep.csv):")
        print(f"  Best Threshold : {best_row['threshold']:.4f}")
        print(f"  Best Utility   : {best_row['utility']:.4f}")
        print(f"  Precision      : {best_row['precision']:.4f}")
        print(f"  Recall         : {best_row['recall']:.4f}")
        print(f"  FPR            : {best_row['fpr']:.4f}")

if __name__ == "__main__":
    main()
