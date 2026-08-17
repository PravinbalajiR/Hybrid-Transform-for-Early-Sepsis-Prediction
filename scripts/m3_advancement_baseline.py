"""
m3_advancement_baseline.py
--------------------------
Phase 0 Baseline Reproduction Script for M3 Advancement.
Verifies exact reproduction of M3 baseline metrics before initiating temporal policy experiments:
  AUROC = 0.961663
  AUPRC = 0.423062
  ECE = 0.0407
  Brier = 0.0213
  Test Utility at th=0.44 (Primary Val Utility Opt): -1.1440
  Test Utility at th=0.60 (Balanced Fallback)     : -0.9535
  Test Utility at th=0.78 (Val F1 Opt)             : -0.8696
"""

import sys
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from evaluation.utility_score import compute_utility_score, threshold_predictions
from evaluation.metrics import compute_ece

RESULTS_DIR = BASE_DIR / "results"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 85)
    print("      M3 ADVANCEMENT PHASE 0 - BASELINE REPRODUCTION AUDIT")
    print("=" * 85)

    ckpt_path = EXPERIMENTS_DIR / "final_m3_frozen" / "best_m3_frozen.pt"
    test_npz_path = RESULTS_DIR / "m3_final_test_predictions.npz"

    print(f"\n1. Artifact SHA256 Checksums:")
    if ckpt_path.exists():
        ckpt_sha = compute_sha256(ckpt_path)
        print(f"   Checkpoint ({ckpt_path.name}) SHA256: {ckpt_sha}")
    else:
        print(f"   Checkpoint ({ckpt_path}) NOT FOUND!")

    if not test_npz_path.exists():
        print(f"   Error: Test NPZ ({test_npz_path}) NOT FOUND!")
        sys.exit(1)

    npz_sha = compute_sha256(test_npz_path)
    print(f"   Prediction NPZ ({test_npz_path.name}) SHA256: {npz_sha}")

    # Load test prediction array
    data = np.load(test_npz_path, allow_pickle=True)
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

    # Compute Continuous Discrimination & Calibration Metrics
    auroc = float(roc_auc_score(y_true_flat, y_proba_flat))
    auprc = float(average_precision_score(y_true_flat, y_proba_flat))
    ece = float(compute_ece(y_true_flat, y_proba_flat))
    brier = float(brier_score_loss(y_true_flat, y_proba_flat))

    print(f"\n2. Continuous Predictive Metrics (N=753,927 hourly records across 20,000 patients):")
    print(f"   AUROC       : {auroc:.6f}  (Expected: 0.961663)")
    print(f"   AUPRC       : {auprc:.6f}  (Expected: 0.423062)")
    print(f"   ECE         : {ece:.4f}      (Expected: 0.0407)")
    print(f"   Brier Score : {brier:.4f}      (Expected: 0.0213)")

    # Verify Utility Scores across Operating Points
    print(f"\n3. Baseline Utility Scores across Locked Operating Points:")
    thresholds = [0.44, 0.60, 0.78]
    expected_utilities = [-1.1440, -0.9535, -0.8696]

    all_passed = True

    for th, exp_u in zip(thresholds, expected_utilities):
        preds = [threshold_predictions(p, th) for p in all_probs]
        u_norm = compute_utility_score(all_labels, preds)
        diff = abs(u_norm - exp_u)
        status = "PASSED" if diff < 1e-3 else "FAILED"
        if diff >= 1e-3:
            all_passed = False
        print(f"   At th = {th:.2f} --> Test Utility: {u_norm:+.4f} (Expected: {exp_u:+.4f}) [{status}]")

    print("\n" + "=" * 85)
    if all_passed and abs(auroc - 0.961663) < 1e-4:
        print("   PHASE 0 BASELINE VERIFICATION PASSED - READY FOR M3-TAP ADVANCEMENT")
    else:
        print("   ERROR: BASELINE METRIC MISMATCH! STOP EXPERIMENT IMMEDIATELY.")
        sys.exit(1)
    print("=" * 85)

if __name__ == "__main__":
    main()
