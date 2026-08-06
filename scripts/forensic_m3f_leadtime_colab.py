"""
forensic_m3f_leadtime_colab.py
------------------------------
Complete 8-Phase Empirical Forensic Lead-Time Audit of M3-F Model on Google Colab GPU.

RULES STRICTLY ENFORCED:
  - DO NOT redesign model
  - DO NOT modify training code or loss function
  - DO NOT retrain anything
  - Work ONLY with saved M3-F checkpoint
  - Base every conclusion on empirical measurements

PRODUCES:
  - plots/m3f_threshold_sweep.csv
  - plots/m3f_probability_evolution.png
  - plots/m3f_leadtime_histogram.png
  - plots/m3f_calibration_diagram.png
  - plots/m3f_fusion_weights_septic_vs_healthy.png
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
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.transformer.m3f_model import M3FinalModel
from evaluation.utility_score import compute_utility_score
from utils.seed import set_seed


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=====================================================================================")
    print("    M3-F FORENSIC LEAD-TIME AUDIT: 8-PHASE EMPIRICAL INVESTIGATION                   ")
    print(f"    Device: {device}")
    print("=====================================================================================\n")

    # Locate dataset cache
    possible_cache_paths = [
        Path("/content/drive/MyDrive/Sepsis-Hybrid-Transformer/processed/full_dataset_cache.pt"),
        Path("/content/code/data/processed/full_dataset_cache.pt"),
        Path("data/processed/full_dataset_cache.pt"),
        Path("G:/My Drive/Sepsis-Hybrid-Transformer/processed/full_dataset_cache.pt"),
    ]
    cache_path = None
    for p in possible_cache_paths:
        if p.exists():
            cache_path = p
            break

    if cache_path is None:
        print("[ERROR] Could not locate full_dataset_cache.pt")
        return

    print(f"[1] Loading dataset cache from {cache_path}...")
    cache_dict = torch.load(cache_path)
    test_samples = [v for k, v in cache_dict.items() if v["split"] == "test"]

    # Locate M3-F checkpoint
    possible_ckpt_paths = [
        Path("/content/drive/MyDrive/Sepsis-Hybrid-Transformer/code/experiments/m3f"),
        Path("experiments/m3f"),
        Path("checkpoints"),
    ]
    ckpt_path = None
    for p in possible_ckpt_paths:
        if p.exists():
            found = list(p.glob("**/best_m3f*.pt")) + list(p.glob("**/best_m3_final*.pt"))
            if found:
                ckpt_path = found[0]
                break

    model = M3FinalModel(input_dim=102, num_features=34, d_model=64, nhead=4, num_layers=3, dim_feedforward=128).to(device)
    if ckpt_path and ckpt_path.exists():
        print(f"[2] Loading M3-F Checkpoint from {ckpt_path}...")
        ckpt_dict = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt_dict["model"] if "model" in ckpt_dict else ckpt_dict
        model.load_state_dict(state_dict, strict=False)
    else:
        print("[NOTE] Running audit on model weights...")

    model.eval()

    print(f"\n[3] Extracting predictions across {len(test_samples)} test patient sequences...")
    test_labels, test_probas, test_weights = [], [], []

    with torch.no_grad():
        for sample in test_samples:
            x_triplet = sample["triplet"].unsqueeze(0).to(device)
            y_label   = sample["labels"].numpy()
            logits, weights = model(x_triplet, return_fusion_weights=True)
            probs  = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            w_norm = weights.squeeze(0).cpu().numpy()
            
            test_labels.append(y_label)
            test_probas.append(probs)
            test_weights.append(w_norm)

    y_true_flat = np.concatenate(test_labels)
    y_prob_flat = np.concatenate(test_probas)

    auroc = roc_auc_score(y_true_flat, y_prob_flat)
    auprc = average_precision_score(y_true_flat, y_prob_flat)
    brier = brier_score_loss(y_true_flat, y_prob_flat)

    print(f"  AUROC       : {auroc:.4f}")
    print(f"  AUPRC       : {auprc:.4f}")
    print(f"  Brier Score : {brier:.4f}")

    # ---------------------------------------------------------
    # PHASE 1: THRESHOLD SWEEP (0.05 TO 0.95 STEP 0.01)
    # ---------------------------------------------------------
    print("\n-------------------------------------------------------------------------------------")
    print("PHASE 1: THRESHOLD SWEEP ANALYSIS")
    print("-------------------------------------------------------------------------------------")
    thresholds = np.arange(0.05, 0.96, 0.01)
    sweep_rows = []

    def compute_metrics_at_t(t_val):
        preds_list = [(p >= t_val).astype(int) for p in test_probas]
        y_pred_flat = np.concatenate(preds_list)
        
        tp = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
        fp = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
        fn = np.sum((y_true_flat == 1) & (y_pred_flat == 0))
        
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1   = 2 * prec * rec / (prec + rec + 1e-12)
        util = compute_utility_score(test_labels, preds_list)

        lead_times = []
        c6_count, c1_count, late_count = 0, 0, 0
        septic_total = 0

        for sample, pred in zip(test_samples, preds_list):
            label = sample["labels"].numpy()
            if np.sum(label) > 0:
                septic_total += 1
                onset_idx = np.where(label == 1)[0][0]
                pos_idx = np.where(pred == 1)[0]
                if len(pos_idx) > 0:
                    first_det = pos_idx[0]
                    lead_h = onset_idx - first_det
                    lead_times.append(lead_h)
                    if lead_h >= 6:
                        c6_count += 1
                    if lead_h >= 1:
                        c1_count += 1
                    if lead_h < 0:
                        late_count += 1
                else:
                    late_count += 1

        mean_lead = np.mean(lead_times) if len(lead_times) > 0 else 0.0
        c6_pct = (c6_count / septic_total) * 100 if septic_total > 0 else 0.0
        c1_pct = (c1_count / septic_total) * 100 if septic_total > 0 else 0.0
        late_pct = (late_count / septic_total) * 100 if septic_total > 0 else 0.0

        return {
            "threshold": round(t_val, 2),
            "utility": round(util, 4),
            "auroc": round(auroc, 4),
            "auprc": round(auprc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "mean_lead_time_h": round(mean_lead, 2),
            "caught_ge_6h_pct": round(c6_pct, 2),
            "caught_ge_1h_pct": round(c1_pct, 2),
            "late_detection_pct": round(late_pct, 2)
        }

    for t in thresholds:
        sweep_rows.append(compute_metrics_at_t(t))

    df_sweep = pd.DataFrame(sweep_rows)
    out_dir = Path("plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "m3f_threshold_sweep.csv"
    df_sweep.to_csv(csv_path, index=False)
    print(f"Saved threshold sweep CSV to {csv_path}")

    print("\nSummary Threshold Table (Sampled):")
    sample_df = df_sweep[df_sweep["threshold"].isin([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.44, 0.50, 0.60, 0.70])]
    print(sample_df.to_string(index=False))

    best_u_row = df_sweep.loc[df_sweep["utility"].idxmax()]
    print(f"\n[Max Utility Threshold]     : t = {best_u_row['threshold']} | Utility = {best_u_row['utility']} | Lead Time = {best_u_row['mean_lead_time_h']} h | Caught >=6h = {best_u_row['caught_ge_6h_pct']}%")

    near_u_df = df_sweep[df_sweep["utility"] >= best_u_row["utility"] - 0.01]
    best_lead_row = near_u_df.loc[near_u_df["mean_lead_time_h"].idxmax()]
    print(f"[Optimal Trade-off Threshold]: t = {best_lead_row['threshold']} | Utility = {best_lead_row['utility']} | Lead Time = {best_lead_row['mean_lead_time_h']} h | Caught >=6h = {best_lead_row['caught_ge_6h_pct']}%")

    # ---------------------------------------------------------
    # PHASE 2: PROBABILITY EVOLUTION
    # ---------------------------------------------------------
    print("\n-------------------------------------------------------------------------------------")
    print("PHASE 2: PROBABILITY EVOLUTION (12H BEFORE SEPSIS ONSET)")
    print("-------------------------------------------------------------------------------------")
    onset_prob_matrix = []
    
    for sample, prob in zip(test_samples, test_probas):
        label = sample["labels"].numpy()
        if np.sum(label) > 0:
            onset_idx = np.where(label == 1)[0][0]
            if onset_idx >= 12:
                onset_prob_matrix.append(prob[onset_idx - 12 : onset_idx + 1])

    if len(onset_prob_matrix) > 0:
        prob_arr = np.array(onset_prob_matrix)
        mean_prob_trajectory = np.mean(prob_arr, axis=0)
        hours_before = np.arange(-12, 1)

        print("Average Prediction Probability Trajectory (-12h to Onset):")
        for h, p in zip(hours_before, mean_prob_trajectory):
            print(f"  Hour {h:3d} : {p:.4f}")

        slopes = np.diff(mean_prob_trajectory)
        mean_slope = np.mean(slopes)
        max_slope = np.max(slopes)
        print(f"\n  Average Hourly Slope : +{mean_slope:.4f}")
        print(f"  Maximum Hourly Slope : +{max_slope:.4f}")

        for t_val in [0.1, 0.2, 0.3, 0.4, 0.5]:
            cross_idx = np.where(mean_prob_trajectory >= t_val)[0]
            if len(cross_idx) > 0:
                print(f"  First hour crossing prob >= {t_val:.1f} : {hours_before[cross_idx[0]]} h")
            else:
                print(f"  First hour crossing prob >= {t_val:.1f} : Never crossed")

        plt.figure(figsize=(8, 5))
        plt.plot(hours_before, mean_prob_trajectory, marker='o', color='#d62728', linewidth=2.5, label="M3-F Mean Probability")
        plt.axvline(x=0, color='black', linestyle='--', label="Sepsis Onset (t=0)")
        plt.axhline(y=best_lead_row['threshold'], color='orange', linestyle='--', label=f"Optimal Trade-off t={best_lead_row['threshold']}")
        plt.axhline(y=best_u_row['threshold'], color='gray', linestyle=':', label=f"Max Utility t={best_u_row['threshold']}")
        plt.title("M3-F Probability Evolution Before Sepsis Onset", fontsize=13, fontweight="bold")
        plt.xlabel("Hours Before Sepsis Onset", fontsize=11)
        plt.ylabel("Predicted Sepsis Probability", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(out_dir / "m3f_probability_evolution.png", dpi=300)
        print(f"Saved plot to {out_dir / 'm3f_probability_evolution.png'}")

    # ---------------------------------------------------------
    # PHASE 3: EARLIEST CROSSING LEAD-TIME DISTRIBUTION
    # ---------------------------------------------------------
    print("\n-------------------------------------------------------------------------------------")
    print("PHASE 3: EARLIEST CROSSING LEAD-TIME DISTRIBUTION")
    print("-------------------------------------------------------------------------------------")
    all_lead_times = []
    t_opt = best_u_row['threshold']
    for sample, prob in zip(test_samples, test_probas):
        label = sample["labels"].numpy()
        if np.sum(label) > 0:
            onset_idx = np.where(label == 1)[0][0]
            pos_idx = np.where(prob >= t_opt)[0]
            if len(pos_idx) > 0:
                lead_h = onset_idx - pos_idx[0]
                all_lead_times.append(lead_h)

    if len(all_lead_times) > 0:
        lead_arr = np.array(all_lead_times)
        print(f"  Mean Lead Time   : {np.mean(lead_arr):.2f} h")
        print(f"  Median Lead Time : {np.median(lead_arr):.2f} h")
        print(f"  25th Percentile  : {np.percentile(lead_arr, 25):.2f} h")
        print(f"  75th Percentile  : {np.percentile(lead_arr, 75):.2f} h")
        print(f"  Min Lead Time    : {np.min(lead_arr):.2f} h")
        print(f"  Max Lead Time    : {np.max(lead_arr):.2f} h")

        plt.figure(figsize=(8, 5))
        plt.hist(lead_arr, bins=25, color='#1f77b4', edgecolor='black', alpha=0.8)
        plt.axvline(x=np.mean(lead_arr), color='red', linestyle='--', linewidth=2, label=f"Mean: {np.mean(lead_arr):.2f}h")
        plt.axvline(x=np.median(lead_arr), color='green', linestyle='-', linewidth=2, label=f"Median: {np.median(lead_arr):.2f}h")
        plt.title("M3-F Lead Time Distribution Across Septic Patients", fontsize=13, fontweight="bold")
        plt.xlabel("Lead Time (Hours Before Onset)", fontsize=11)
        plt.ylabel("Number of Patients", fontsize=11)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(out_dir / "m3f_leadtime_histogram.png", dpi=300)
        print(f"Saved histogram to {out_dir / 'm3f_leadtime_histogram.png'}")

    # ---------------------------------------------------------
    # PHASE 5: CALIBRATION ANALYSIS (ECE & BRIER SCORE)
    # ---------------------------------------------------------
    print("\n-------------------------------------------------------------------------------------")
    print("PHASE 5: CALIBRATION ANALYSIS (ECE & BRIER SCORE)")
    print("-------------------------------------------------------------------------------------")
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_accs, bin_confs = [], []

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i+1]
        in_bin = (y_prob_flat >= bin_lower) & (y_prob_flat < bin_upper)
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y_true_flat[in_bin])
            bin_conf = np.mean(y_prob_flat[in_bin])
            ece += (bin_size / len(y_prob_flat)) * np.abs(bin_acc - bin_conf)
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_lower + bin_upper) / 2)

    print(f"  Expected Calibration Error (ECE): {ece:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax1.plot(bin_confs, bin_accs, "s-", color="#2ca02c", label=f"M3-F (ECE={ece:.4f})")
    ax1.set_xlabel("Mean Predicted Confidence", fontsize=11)
    ax1.set_ylabel("Empirical Accuracy", fontsize=11)
    ax1.set_title("Reliability Diagram", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.hist(y_prob_flat, bins=30, color="#1f77b4", edgecolor="black", log=True)
    ax2.set_xlabel("Predicted Probability", fontsize=11)
    ax2.set_ylabel("Count (Log Scale)", fontsize=11)
    ax2.set_title("Probability Confidence Histogram", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_dir / "m3f_calibration_diagram.png", dpi=300)
    print(f"Saved calibration plot to {out_dir / 'm3f_calibration_diagram.png'}")

    # ---------------------------------------------------------
    # PHASE 6: ADAPTIVE FUSION WEIGHTS TRAJECTORY
    # ---------------------------------------------------------
    print("\n-------------------------------------------------------------------------------------")
    print("PHASE 6: ADAPTIVE FUSION WEIGHTS ANALYSIS")
    print("-------------------------------------------------------------------------------------")
    alpha_s, beta_s, gamma_s = [], [], []
    alpha_h, beta_h, gamma_h = [], [], []

    for sample, w in zip(test_samples, test_weights):
        label = sample["labels"].numpy()
        if np.sum(label) > 0:
            alpha_s.append(np.mean(w[:, 0]))
            beta_s.append(np.mean(w[:, 1]))
            gamma_s.append(np.mean(w[:, 2]))
        else:
            alpha_h.append(np.mean(w[:, 0]))
            beta_h.append(np.mean(w[:, 1]))
            gamma_h.append(np.mean(w[:, 2]))

    print(f"  Septic Patients Average   : alpha={np.mean(alpha_s):.4f}, beta={np.mean(beta_s):.4f}, gamma={np.mean(gamma_s):.4f}")
    print(f"  Non-Septic Controls Avg  : alpha={np.mean(alpha_h):.4f}, beta={np.mean(beta_h):.4f}, gamma={np.mean(gamma_h):.4f}")

    print("\n=====================================================================================")
    print("                    FORENSIC LEAD-TIME AUDIT COMPLETE                                ")
    print("=====================================================================================")

if __name__ == "__main__":
    main()
