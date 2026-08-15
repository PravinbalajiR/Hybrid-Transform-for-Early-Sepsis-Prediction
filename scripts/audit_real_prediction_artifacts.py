"""
audit_real_prediction_artifacts.py
----------------------------------
Gate B.1 Provenance Audit Script: Inspects real NPZ prediction artifacts in results/
to trace exact y_true and y_proba distributions for M3 and evaluate AUROC/AUPRC.
"""

import sys
import json
import torch
import numpy as np
import pandas as pd
import hashlib
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from scipy.integrate import trapezoid

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

def main():
    print("=" * 80)
    print("   GATE B.1 — REAL PREDICTION ARTIFACT PROVENANCE AUDIT")
    print("=" * 80)

    npz_path = RESULTS_DIR / "m3_final_test_predictions.npz"
    thresh_path = RESULTS_DIR / "m3_selected_thresholds.json"

    print(f"\n1. Prediction Artifact Path: {npz_path}")
    print(f"   Exists: {npz_path.exists()}")

    if not npz_path.exists():
        print("   ERROR: m3_final_test_predictions.npz NOT FOUND!")
        return

    # Check file size & SHA256
    with open(npz_path, "rb") as f:
        file_bytes = f.read()
        file_sha256 = hashlib.sha256(file_bytes).hexdigest()
    print(f"   File Size   : {len(file_bytes):,} bytes")
    print(f"   SHA256 Hash : {file_sha256}")

    # Load NPZ contents
    data = np.load(npz_path, allow_pickle=True)
    print(f"\n2. NPZ Keys Present: {list(data.keys())}")

    y_true = data["y_true_flat"]
    y_proba = data["y_proba_flat"]

    print("\n3. Array Dimensions & Label Distribution:")
    print(f"   Total Observations (N) : {len(y_true):,}")
    print(f"   Positive Labels (1s)   : {y_true.sum():,} ({y_true.mean()*100:.2f}%)")
    print(f"   Negative Labels (0s)   : {(1-y_true).sum():,} ({(1-y_true.mean())*100:.2f}%)")

    print("\n4. Prediction Probability Distribution:")
    print(f"   Min Probability  : {y_proba.min():.6f}")
    print(f"   Max Probability  : {y_proba.max():.6f}")
    print(f"   Mean Probability : {y_proba.mean():.6f}")
    print(f"   Std Probability  : {y_proba.std():.6f}")

    # Evaluate AUROC and AUPRC on REAL saved test arrays
    auroc = roc_auc_score(y_true, y_proba)
    auprc = average_precision_score(y_true, y_proba)

    print("\n5. Empirical Metrics on Real Saved NPZ Predictions:")
    print(f"   AUROC (scikit-learn) : {auroc:.6f}")
    print(f"   AUPRC (scikit-learn) : {auprc:.6f}")

    if thresh_path.exists():
        selected_thresh = json.loads(thresh_path.read_text())
        print(f"\n6. Selected Thresholds File: {selected_thresh}")

    print("\n" + "=" * 80)
    print(f"   PROVENANCE VERDICT: REAL M3 SAVED NPZ PRODUCES EXACTLY AUROC = {auroc:.4f}, AUPRC = {auprc:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
