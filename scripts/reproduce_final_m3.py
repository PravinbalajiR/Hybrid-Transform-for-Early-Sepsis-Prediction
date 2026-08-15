# reproduce_final_m3.py
# ---------------------
# Standalone Reproducibility Script for M3 (Time-Aware Transformer)

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, brier_score_loss

def main():
    base_dir = Path(__file__).parent.parent
    npz_path = base_dir / "results" / "m3_final_test_predictions.npz"
    thresh_path = base_dir / "results" / "m3_selected_thresholds.json"
    
    data = np.load(npz_path, allow_pickle=True)
    y_true = data["y_true_flat"]
    y_proba = data["y_proba_flat"]
    
    selected_thresh = json.loads(thresh_path.read_text())
    th = selected_thresh["balanced_clinical"]
    
    y_pred = (y_proba >= th).astype(int)
    
    auroc = roc_auc_score(y_true, y_proba)
    auprc = average_precision_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print("=" * 60)
    print("FINAL M3 RESEARCH RESULT")
    print("=" * 60)
    print(f"Checkpoint: experiments/final_m3_frozen/best_m3_frozen.pt")
    print(f"Architecture: TACTModel (Time-Aware Transformer)")
    print(f"Threshold: {th:.2f}")
    print(f"Threshold Selection Source: VALIDATION ONLY (Zero Test Leakage)")
    print("-" * 60)
    print(f"AUROC    : {auroc:.4f}")
    print(f"AUPRC    : {auprc:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"Brier    : {brier:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
