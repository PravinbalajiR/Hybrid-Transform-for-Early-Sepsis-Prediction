"""
evaluate_historical_run.py
--------------------------
Evaluates the historical checkpoint:
  run_20260802_073034 / best_time_aware_transformer_auroc0.973_epoch25.pt
(which achieved AUROC = 0.9697, AUPRC = 0.4851, Lead Time = 5.65h, Utility = -0.9932)

Directly evaluates the exact saved model weights on real test data under BOTH:
  (a) Old Evaluator (fixed threshold t = 0.50, un-normalized utility)
  (b) Official Evaluator (optimal t_opt on val set, PhysioNet 2019 normalized utility)

WITHOUT RETRAINING OR MODIFYING ANY SOURCE CODE.
"""

from __future__ import annotations

import sys
import os
import math
from pathlib import Path
import json
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.utility_score import compute_utility_score as compute_utility_new
from utils.seed import set_seed


def compute_utility_old(labels_list, preds_list):
    total_u = 0.0
    for y_true, y_pred in zip(labels_list, preds_list):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        raw_u = 1.0 * tp - 0.05 * fp - 2.0 * fn
        total_u += raw_u
    return total_u / len(labels_list)


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=====================================================================================")
    print("    HISTORICAL REPRODUCTION EVALUATION: CHECKPOINT epoch25 (run_20260802_073034)      ")
    print(f"    Running on Device: {device}")
    print("=====================================================================================\n")

    # Path resolution for Google Colab or local environment
    possible_ckpt_paths = [
        Path("/content/drive/MyDrive/Sepsis-Hybrid-Transformer/code/experiments/time_aware_transformer/run_20260802_073034/checkpoints/best_time_aware_transformer_auroc0.973_epoch25.pt"),
        Path("G:/My Drive/Sepsis-Hybrid-Transformer/code/experiments/time_aware_transformer/run_20260802_073034/checkpoints/best_time_aware_transformer_auroc0.973_epoch25.pt"),
        Path("experiments/time_aware_transformer/run_20260802_073034/checkpoints/best_time_aware_transformer_auroc0.973_epoch25.pt"),
    ]

    ckpt_path = None
    for p in possible_ckpt_paths:
        if p.exists():
            ckpt_path = p
            break

    if ckpt_path is None:
        print("[ERROR] Historical checkpoint not found in standard paths.")
        return

    print(f"[1] Loading historical checkpoint: {ckpt_path}")
    checkpoint_dict = torch.load(ckpt_path, map_location=device)

    # Initialize exact historical model architecture (Linear 102 -> 64 without Time2Vec)
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model=64, max_len=500):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0))
        def forward(self, x):
            return x + self.pe[:, :x.size(1)]

    class LegacyTimeAwareEmbedding(nn.Module):
        def __init__(self, input_dim=102, d_model=64):
            super().__init__()
            self.proj = nn.Linear(102, d_model)
            self.layer_norm = nn.LayerNorm(d_model)
            self.pos_encoder = PositionalEncoding(d_model)
        def forward(self, x):
            out = self.layer_norm(self.proj(x) * math.sqrt(64))
            return self.pos_encoder(out)

    class LegacyTACTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = LegacyTimeAwareEmbedding(102, 64)
            encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
            self.fc_out = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1)
            )
        def forward(self, x):
            emb = self.embedding(x)
            out = self.transformer_encoder(emb)
            return self.fc_out(out)

    model = LegacyTACTModel().to(device)
    if "model" in checkpoint_dict:
        model.load_state_dict(checkpoint_dict["model"])
    else:
        model.load_state_dict(checkpoint_dict)
    model.eval()

    possible_cache_paths = [
        Path("/content/drive/MyDrive/Sepsis-Hybrid-Transformer/processed/full_dataset_cache.pt"),
        Path("G:/My Drive/Sepsis-Hybrid-Transformer/processed/full_dataset_cache.pt"),
        Path("data/processed/full_dataset_cache.pt"),
    ]
    cache_path = None
    for p in possible_cache_paths:
        if p.exists():
            cache_path = p
            break

    print(f"[2] Loading real patient dataset from {cache_path}...")
    cache_dict = torch.load(cache_path)
    val_samples  = [v for k, v in cache_dict.items() if v["split"] == "val"]
    test_samples = [v for k, v in cache_dict.items() if v["split"] == "test"]

    def predict_split(samples):
        labels_list = []
        probas_list = []
        with torch.no_grad():
            for sample in samples:
                x_triplet = sample["triplet"].unsqueeze(0).to(device)
                y_label   = sample["labels"].numpy()
                logits = model(x_triplet).squeeze(0).squeeze(-1)
                probs  = torch.sigmoid(logits).cpu().numpy()
                labels_list.append(y_label)
                probas_list.append(probs)
        return labels_list, probas_list

    val_labels, val_probas   = predict_split(val_samples)
    test_labels, test_probas = predict_split(test_samples)

    y_true_flat = np.concatenate(test_labels)
    y_prob_flat = np.concatenate(test_probas)

    # ---------------------------------------------------------
    # EVALUATION A: Old Evaluator (Fixed t = 0.5, Old Utility)
    # ---------------------------------------------------------
    t_old = 0.50
    preds_old = [(p >= t_old).astype(int) for p in test_probas]
    y_pred_old_flat = np.concatenate(preds_old)

    auroc = roc_auc_score(y_true_flat, y_prob_flat)
    auprc = average_precision_score(y_true_flat, y_prob_flat)

    tp_o = np.sum((y_true_flat == 1) & (y_pred_old_flat == 1))
    fp_o = np.sum((y_true_flat == 0) & (y_pred_old_flat == 1))
    fn_o = np.sum((y_true_flat == 1) & (y_pred_old_flat == 0))
    prec_o = tp_o / (tp_o + fp_o + 1e-12)
    rec_o  = tp_o / (tp_o + fn_o + 1e-12)
    f1_o   = 2 * prec_o * rec_o / (prec_o + rec_o + 1e-12)
    u_old  = compute_utility_old(test_labels, preds_old)
    u_norm_at_old = compute_utility_new(test_labels, preds_old)

    # ---------------------------------------------------------
    # EVALUATION B: Current Official Evaluator (Optimal t_opt on Val, Official Utility)
    # ---------------------------------------------------------
    thresholds = np.linspace(0.01, 0.99, 50)
    best_t = 0.5
    best_val_u = -np.inf
    for t in thresholds:
        val_preds_t = [(p >= t).astype(int) for p in val_probas]
        u_t = compute_utility_new(val_labels, val_preds_t)
        if u_t > best_val_u:
            best_val_u = u_t
            best_t = t

    preds_new = [(p >= best_t).astype(int) for p in test_probas]
    y_pred_new_flat = np.concatenate(preds_new)

    tp_n = np.sum((y_true_flat == 1) & (y_pred_new_flat == 1))
    fp_n = np.sum((y_true_flat == 0) & (y_pred_new_flat == 1))
    fn_n = np.sum((y_true_flat == 1) & (y_pred_new_flat == 0))
    prec_n = tp_n / (tp_n + fp_n + 1e-12)
    rec_n  = tp_n / (tp_n + fn_n + 1e-12)
    f1_n   = 2 * prec_n * rec_n / (prec_n + rec_n + 1e-12)
    u_official = compute_utility_new(test_labels, preds_new)

    def compute_lead_times(samples, preds_list):
        lead_times = []
        for sample, pred in zip(samples, preds_list):
            label = sample["labels"].numpy()
            if np.sum(label) > 0:
                onset_idx = np.where(label == 1)[0][0]
                pos_idx = np.where(pred == 1)[0]
                if len(pos_idx) > 0:
                    first_det = pos_idx[0]
                    lead_h = onset_idx - first_det
                    lead_times.append(lead_h)
        return np.mean(lead_times) if len(lead_times) > 0 else 0.0

    lead_old = compute_lead_times(test_samples, preds_old)
    lead_new = compute_lead_times(test_samples, preds_new)

    print("=====================================================================================")
    print("                     FORENSIC COMPARISON TABLE (SAME CHECKPOINT)                     ")
    print("=====================================================================================")
    print(f"| Metric            | (a) Old Evaluator (t=0.50) | (b) Official Evaluator (t={best_t:.2f}) | Delta |")
    print(f"| ----------------- | ------------------------- | ------------------------------ | ----- |")
    print(f"| AUROC             | {auroc:25.4f} | {auroc:30.4f} | +0.0000 |")
    print(f"| AUPRC             | {auprc:25.4f} | {auprc:30.4f} | +0.0000 |")
    print(f"| Precision         | {prec_o:25.4f} | {prec_n:30.4f} | {prec_n-prec_o:+0.4f} |")
    print(f"| Recall            | {rec_o:25.4f} | {rec_n:30.4f} | {rec_n-rec_o:+0.4f} |")
    print(f"| F1 Score          | {f1_o:25.4f} | {f1_n:30.4f} | {f1_n-f1_o:+0.4f} |")
    print(f"| Mean Lead Time (h)| {lead_old:25.2f} | {lead_new:30.2f} | {lead_new-lead_old:+0.2f} |")
    print(f"| Utility Score     | {u_old:25.4f} (Unnorm)    | {u_official:30.4f} (PhysioNet)  | N/A |")
    print(f"| Utility (PhysioNet| {u_norm_at_old:25.4f}            | {u_official:30.4f}              | {u_official-u_norm_at_old:+0.4f} |")
    print("=====================================================================================\n")

if __name__ == "__main__":
    main()
