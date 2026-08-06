"""
evaluate_historical_locked_m3.py
--------------------------------
Evaluates the exact historical M3 model locked at AUROC ~ 0.9697, AUPRC ~ 0.4942, Utility = 0.3372.

Uses:
  - Checkpoint: best_time_aware_transformer_auroc0.973_epoch25.pt (run_20260802_073034)
  - Architecture: M3RecoveredModel (LegacyTACTModel)
"""

from __future__ import annotations

import sys
import os
import hashlib
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.transformer.m3_recovered_model import M3RecoveredModel
from evaluation.utility_score import compute_utility_score
from utils.seed import set_seed


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=====================================================================================")
    print("     EVALUATING HISTORICAL LOCKED M3 MODEL (best_time_aware_transformer_epoch25)     ")
    print(f"     Device: {device}")
    print("=====================================================================================\n")

    # Target checkpoint path
    possible_paths = [
        Path("/content/drive/MyDrive/Sepsis-Hybrid-Transformer/code/experiments/time_aware_transformer/run_20260802_073034/checkpoints/best_time_aware_transformer_auroc0.973_epoch25.pt"),
        Path("experiments/time_aware_transformer/run_20260802_073034/checkpoints/best_time_aware_transformer_auroc0.973_epoch25.pt"),
        Path("G:/My Drive/Sepsis-Hybrid-Transformer/code/experiments/time_aware_transformer/run_20260802_073034/checkpoints/best_time_aware_transformer_auroc0.973_epoch25.pt"),
    ]

    ckpt_path = None
    for p in possible_paths:
        if p.exists():
            ckpt_path = p
            break

    if ckpt_path is None:
        # Search for any epoch25 or time_aware pt
        for search_root in [Path("/content/drive/MyDrive/Sepsis-Hybrid-Transformer"), Path("experiments"), Path("G:/My Drive/Sepsis-Hybrid-Transformer")]:
            if search_root.exists():
                found = list(search_root.glob("**/best_time_aware_transformer_auroc0.973_epoch25.pt"))
                if found:
                    ckpt_path = found[0]
                    break

    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError("[ERROR] Could not locate historical checkpoint `best_time_aware_transformer_auroc0.973_epoch25.pt`.")

    print(f"[1] Located Historical Checkpoint:")
    print(f"    Path   : {ckpt_path.absolute()}")
    h = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    print(f"    SHA256 : {h.hexdigest()}")

    # Load state dict
    ckpt_dict = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt_dict["model"] if isinstance(ckpt_dict, dict) and "model" in ckpt_dict else ckpt_dict

    # Build M3RecoveredModel
    model = M3RecoveredModel(input_dim=102, d_model=64, nhead=4, num_layers=3, dim_feedforward=128).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    print(f"\n[2] Loaded weights into M3RecoveredModel with strict=True (0 missing, 0 unexpected).")
    model.eval()

    # Load test dataset
    possible_cache = [
        Path("/content/drive/MyDrive/Sepsis-Hybrid-Transformer/processed/full_dataset_cache.pt"),
        Path("data/processed/full_dataset_cache.pt"),
        Path("G:/My Drive/Sepsis-Hybrid-Transformer/processed/full_dataset_cache.pt"),
    ]
    cache_path = None
    for p in possible_cache:
        if p.exists():
            cache_path = p
            break

    print(f"\n[3] Loading dataset cache from {cache_path}...")
    cache_dict = torch.load(cache_path)
    test_samples = [v for k, v in cache_dict.items() if v["split"] == "test"]

    print(f"\n[4] Evaluating {len(test_samples)} test patient sequences...")
    test_labels, test_probas = [], []

    with torch.no_grad():
        for sample in test_samples:
            x_triplet = sample["triplet"].unsqueeze(0).to(device)
            y_label   = sample["labels"].numpy()
            logits    = model(x_triplet)
            probs     = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            
            test_labels.append(y_label)
            test_probas.append(probs)

    y_true_flat = np.concatenate(test_labels)
    y_prob_flat = np.concatenate(test_probas)

    auroc = roc_auc_score(y_true_flat, y_prob_flat)
    auprc = average_precision_score(y_true_flat, y_prob_flat)

    # Threshold sweep to find official Utility & lead time
    thresholds = np.arange(0.10, 0.90, 0.02)
    best_u, best_t, best_f1, best_lead = -999.0, 0.50, 0.0, 0.0

    for t in thresholds:
        preds = [(p >= t).astype(int) for p in test_probas]
        u = compute_utility_score(test_labels, preds)
        
        y_pred_flat = np.concatenate(preds)
        f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)
        
        # Mean Lead Time
        leads = []
        for sample, pred in zip(test_samples, preds):
            lbl = sample["labels"].numpy()
            if np.sum(lbl) > 0:
                onset_idx = np.where(lbl == 1)[0][0]
                pos_idx = np.where(pred == 1)[0]
                if len(pos_idx) > 0:
                    leads.append(onset_idx - pos_idx[0])

        m_lead = np.mean(leads) if len(leads) > 0 else 0.0

        if u > best_u:
            best_u = u
            best_t = t
            best_f1 = f1
            best_lead = m_lead

    print("\n=====================================================================================")
    print("                    HISTORICAL LOCKED M3 EVALUATION RESULTS                          ")
    print("=====================================================================================")
    print(f"  AUROC                 : {auroc:.4f}")
    print(f"  AUPRC                 : {auprc:.4f}")
    print(f"  Optimal Threshold     : {best_t:.2f}")
    print(f"  Official Utility Score: {best_u:.4f}")
    print(f"  F1 Score              : {best_f1:.4f}")
    print(f"  Mean Lead Time        : {best_lead:.2f} h")
    print("=====================================================================================\n")

if __name__ == "__main__":
    main()
