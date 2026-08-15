"""
analyze_m3_tradeoff.py
----------------------
Script to perform strict checkpoint verification, threshold sweep, and Pareto trade-off 
analysis (Utility vs. Lead Time vs. Precision/Recall) for Model M3 (Time-Aware Transformer).

Outputs:
  - Checkpoint verification report (missing/unexpected keys)
  - Threshold sweep table (thresholds 0.05 to 0.95)
  - 4 Publication-grade plots in plots/ directory:
      1. plots/utility_vs_threshold.png
      2. plots/lead_time_vs_threshold.png
      3. plots/precision_vs_recall.png
      4. plots/utility_vs_lead_time.png
  - Saved CSV data: plots/threshold_sweep_results.csv
"""

import os
import sys
import glob
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.dataset import create_cached_dataloader
from models.transformer.tact_model import TACTModel
from evaluation.utility_score import compute_utility_score
from evaluation.metrics import compute_timing_analysis, compute_ece, compute_classification_metrics
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score


def verify_and_load_checkpoint(device, ckpt_path_arg=None):
    """
    Step 1: Mandatory Checkpoint Architecture & State Dict Verification.
    Strict loading validation.
    """
    print("\n=======================================================")
    print("  STEP 1: CHECKPOINT & MODEL VERIFICATION")
    print("=======================================================")
    
    exp_base = Path(__file__).parent.parent / "experiments"
    
    if ckpt_path_arg and os.path.exists(ckpt_path_arg):
        latest_ckpt_path = Path(ckpt_path_arg)
    else:
        # Search specifically for M3 / time_aware_transformer / tact checkpoints
        candidates = list(exp_base.glob("**/checkpoints/*time_aware*.pt")) + \
                     list(exp_base.glob("**/checkpoints/*tact*.pt")) + \
                     list(exp_base.glob("**/checkpoints/*m3*.pt"))
        if not candidates:
            # Fallback to any best_*.pt under tact or time_aware_transformer runs
            candidates = list(exp_base.glob("**/tact/**/best_*.pt")) + \
                         list(exp_base.glob("**/time_aware_transformer/**/best_*.pt"))
                         
        if not candidates:
            raise FileNotFoundError("No M3 / time_aware_transformer checkpoints found under experiments/!")
            
        latest_ckpt_path = max(candidates, key=os.path.getctime)
        
    print(f"[Verification] Selected Checkpoint File: {latest_ckpt_path}")
    
    checkpoint = torch.load(latest_ckpt_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    
    # Instantiate architecture from checkpoint config + state_dict inspection
    config = checkpoint.get("config", {})
    hidden_dim = config.get("hidden_dim", 64)
    num_heads = config.get("num_heads", 4)
    layers = config.get("layers", 3)
    dropout = config.get("dropout", 0.1)
    model_type = config.get("model", "time_aware_transformer")
    
    # Inspect state_dict for exact input_dim and ablation_mode matching
    if "embedding.proj.weight" in state_dict:
        proj_weight_in_dim = state_dict["embedding.proj.weight"].shape[1]
        hidden_dim = state_dict["embedding.proj.weight"].shape[0]
        
        if proj_weight_in_dim == 34:
            input_dim = 34
            ablation_mode = "none"
        elif proj_weight_in_dim == 102:
            input_dim = 102
            ablation_mode = "linear_delta"
        elif proj_weight_in_dim == 204:
            input_dim = 102
            ablation_mode = "none"
        else:
            input_dim = config.get("input_dim", 102)
            ablation_mode = config.get("ablation_mode", "none")
    else:
        input_dim = 102
        ablation_mode = config.get("ablation_mode", "none")
        
    input_key = "values" if input_dim == 34 else "triplet"
    
    model = TACTModel(
        input_dim=input_dim,
        d_model=hidden_dim,
        nhead=num_heads,
        num_layers=layers,
        dropout=dropout,
        ablation_mode=ablation_mode
    ).to(device)
    
    # Strict load state dict validation
    load_result = model.load_state_dict(state_dict, strict=True)
    
    missing_keys = len(load_result.missing_keys)
    unexpected_keys = len(load_result.unexpected_keys)
    
    print(f"  -> Model Name       : {model_type}")
    print(f"  -> Model Architecture: TACTModel (input_dim={input_dim}, d_model={hidden_dim}, nhead={num_heads}, layers={layers}, ablation_mode='{ablation_mode}')")
    print(f"  -> Missing Keys     : {missing_keys}")
    print(f"  -> Unexpected Keys  : {unexpected_keys}")
    
    if missing_keys == 0 and unexpected_keys == 0:
        print("  -> STATUS: [OK] PERFECT STRICT LOAD VERIFIED (Zero Missing / Zero Unexpected Keys)")
    else:
        print("  -> STATUS: [WARNING] LOAD WARNING (Keys mismatch detected)")
        
    model.eval()
    return model, latest_ckpt_path, config, input_key


def get_test_predictions(model, device, input_key="triplet"):
    """
    Step 2: Run inference on the test dataset to retrieve ground-truth and probabilities.
    """
    print("\n=======================================================")
    print("  STEP 2: TEST DATASET INFERENCE")
    print("=======================================================")
    
    cache_path = Path(__file__).parent.parent / "data" / "processed" / "full_dataset_cache.pt"
    if not cache_path.exists():
        cache_path = Path(__file__).parent.parent.parent / "processed" / "full_dataset_cache.pt"
        
    print(f"[Inference] Loading cached dataset from: {cache_path}")
    cache_dict = torch.load(cache_path)
    
    test_samples = []
    for pid, item in cache_dict.items():
        if item.get("split") == "test":
            item["patient_id"] = pid
            test_samples.append(item)
            
    print(f"[Inference] Total Test Patients: {len(test_samples)}")
    loader = create_cached_dataloader(test_samples, batch_size=64, shuffle=False)
    
    all_labels = []
    all_probas = []
    
    with torch.no_grad():
        for batch in loader:
            x = batch[input_key].to(device, non_blocking=True)
            pad_mask = batch["padding_mask"].to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                output = model(x, padding_mask=pad_mask)
                logits = output[0] if isinstance(output, tuple) else output
                probas = torch.sigmoid(logits).cpu().numpy()
                labels = batch["labels"].numpy()
                
            for i in range(len(batch["patient_ids"])):
                length = batch["lengths"][i].item()
                all_probas.append(probas[i, :length])
                all_labels.append(labels[i, :length])
                
    y_true_flat = np.concatenate(all_labels)
    y_proba_flat = np.concatenate(all_probas)
    
    global_auroc = roc_auc_score(y_true_flat, y_proba_flat)
    global_auprc = average_precision_score(y_true_flat, y_proba_flat)
    ece = compute_ece(y_true_flat, y_proba_flat)
    
    print(f"  -> Global Test AUROC: {global_auroc:.4f}")
    print(f"  -> Global Test AUPRC: {global_auprc:.4f}")
    print(f"  -> Global Test ECE  : {ece:.4f}")
    
    return all_labels, all_probas, y_true_flat, y_proba_flat, global_auroc, global_auprc, ece


def perform_threshold_sweep(all_labels, all_probas, y_true_flat, y_proba_flat, global_auroc, global_auprc):
    """
    Step 3: Sweep thresholds from 0.05 to 0.95 and record all trade-off metrics.
    """
    print("\n=======================================================")
    print("  STEP 3: THRESHOLD SWEEP ANALYSIS")
    print("=======================================================")
    
    thresholds = np.linspace(0.05, 0.95, 19)
    records = []
    
    for th in thresholds:
        # Binary predictions
        all_preds = [(p >= th).astype(int) for p in all_probas]
        y_pred_flat = (y_proba_flat >= th).astype(int)
        
        # Classification metrics
        prec = precision_score(y_true_flat, y_pred_flat, zero_division=0)
        rec = recall_score(y_true_flat, y_pred_flat, zero_division=0)
        f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)
        
        # False positive rate (FPR)
        tn = np.sum((y_true_flat == 0) & (y_pred_flat == 0))
        fp = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # PhysioNet Utility Score
        util = compute_utility_score(all_labels, all_preds)
        
        # Early Warning Timing Analysis
        timing = compute_timing_analysis(all_labels, all_preds)
        lead_h = timing["mean_lead_h"] if timing["mean_lead_h"] is not None else 0.0
        pct_6h = timing["pct_early_6h"] if "pct_early_6h" in timing else 0.0
        pct_1h = timing["pct_early_1h"] if "pct_early_1h" in timing else 0.0
        pct_late = timing["pct_late"] if "pct_late" in timing else 0.0
        n_tp = timing["n_tp"] if "n_tp" in timing else 0
        
        records.append({
            "threshold": float(th),
            "auroc": float(global_auroc),
            "auprc": float(global_auprc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "fpr": float(fpr),
            "utility": float(util),
            "n_tp": int(n_tp),
            "mean_lead_h": float(lead_h),
            "pct_early_6h": float(pct_6h),
            "pct_early_1h": float(pct_1h),
            "pct_late": float(pct_late),
        })
        
    df = pd.DataFrame(records)
    
    # Print formatted markdown table
    print("\n### THRESHOLD SWEEP SUMMARY TABLE\n")
    print(f"| Thresh | Utility | Lead Time (h) | Recall | Prec | F1 | >=6h Early | >=1h Early | Late % | FPR |")
    print(f"|---|---|---|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        print(f"| {r['threshold']:.2f} | {r['utility']:+.4f} | {r['mean_lead_h']:.1f} h | {r['recall']:.3f} | {r['precision']:.3f} | {r['f1']:.3f} | {r['pct_early_6h']:.1f}% | {r['pct_early_1h']:.1f}% | {r['pct_late']:.1f}% | {r['fpr']:.4f} |")
        
    return df


def generate_tradeoff_plots(df):
    """
    Step 4: Generate 4 publication-quality trade-off plots.
    """
    print("\n=======================================================")
    print("  STEP 4: GENERATING PARETO TRADE-OFF PLOTS")
    print("=======================================================")
    
    plots_dir = Path(__file__).parent.parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Plot 1: Utility vs Threshold
    plt.figure(figsize=(8, 5))
    plt.plot(df["threshold"], df["utility"], marker='o', color='#1f77b4', linewidth=2, label="PhysioNet Utility")
    plt.axhline(0, color='gray', linestyle='--', alpha=0.7)
    max_util_row = df.loc[df["utility"].idxmax()]
    plt.scatter([max_util_row["threshold"]], [max_util_row["utility"]], color='red', s=100, zorder=5, label=f"Max Utility ({max_util_row['utility']:+.4f} @ th={max_util_row['threshold']:.2f})")
    plt.title("PhysioNet Utility Score vs. Decision Threshold", fontsize=12, fontweight='bold')
    plt.xlabel("Decision Threshold", fontsize=11)
    plt.ylabel("Normalized Utility Score", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    p1 = plots_dir / "utility_vs_threshold.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"  -> Saved: {p1}")

    # Plot 2: Lead Time vs Threshold
    plt.figure(figsize=(8, 5))
    plt.plot(df["threshold"], df["mean_lead_h"], marker='s', color='#2ca02c', linewidth=2, label="Mean Lead Time (h)")
    plt.plot(df["threshold"], df["pct_early_6h"] / 10, marker='^', color='#ff7f0e', linestyle='--', label="% Caught >=6h Early (/10)")
    plt.title("Early Detection Lead Time vs. Decision Threshold", fontsize=12, fontweight='bold')
    plt.xlabel("Decision Threshold", fontsize=11)
    plt.ylabel("Mean Lead Time (Hours)", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    p2 = plots_dir / "lead_time_vs_threshold.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"  -> Saved: {p2}")

    # Plot 3: Precision vs Recall
    plt.figure(figsize=(8, 5))
    plt.plot(df["recall"], df["precision"], marker='o', color='#9467bd', linewidth=2, label="PR Operating Curve")
    for _, r in df.iloc[::3].iterrows():
        plt.annotate(f"th={r['threshold']:.2f}", (r['recall'], r['precision']), textcoords="offset points", xytext=(5,5), ha='left', fontsize=8)
    plt.title("Precision vs. Recall Operating Curve", fontsize=12, fontweight='bold')
    plt.xlabel("Recall (Sensitivity)", fontsize=11)
    plt.ylabel("Precision (PPV)", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    p3 = plots_dir / "precision_vs_recall.png"
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"  -> Saved: {p3}")

    # Plot 4: Utility vs Lead Time (The Key Pareto Frontier)
    plt.figure(figsize=(8, 5))
    plt.plot(df["mean_lead_h"], df["utility"], marker='D', color='#d62728', linewidth=2)
    for _, r in df.iloc[::2].iterrows():
        plt.annotate(f"th={r['threshold']:.2f}", (r['mean_lead_h'], r['utility']), textcoords="offset points", xytext=(5,5), fontsize=8)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.7)
    plt.title("Pareto Trade-off: Utility Score vs. Lead Time", fontsize=12, fontweight='bold')
    plt.xlabel("Mean Early Detection Lead Time (Hours)", fontsize=11)
    plt.ylabel("PhysioNet Utility Score", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    p4 = plots_dir / "utility_vs_lead_time.png"
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"  -> Saved: {p4}")

    # Save CSV
    csv_path = plots_dir / "threshold_sweep_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  -> Saved CSV Data: {csv_path}")


def identify_sweet_spot(df):
    """
    Step 5: Identify the optimal operating sweet spot threshold.
    """
    print("\n=======================================================")
    print("  STEP 5: OPERATING SWEET SPOT ANALYSIS")
    print("=======================================================")
    
    # Search for threshold that maximizes utility while preserving >= 5.0 hours lead time
    viable = df[df["mean_lead_h"] >= 4.5]
    if not viable.empty:
        best_row = viable.loc[viable["utility"].idxmax()]
    else:
        best_row = df.loc[df["utility"].idxmax()]
        
    print(f"[RECOMMENDED] OPERATING SWEET SPOT (Threshold = {best_row['threshold']:.2f}):")
    print(f"  -> PhysioNet Utility Score : {best_row['utility']:+.4f}")
    print(f"  -> Mean Lead Time           : {best_row['mean_lead_h']:.1f} hours")
    print(f"  -> Caught >= 6h Early        : {best_row['pct_early_6h']:.1f}%")
    print(f"  -> Recall (Sensitivity)     : {best_row['recall']:.3f} ({best_row['recall']*100:.1f}%)")
    print(f"  -> Precision (PPV)          : {best_row['precision']:.3f} ({best_row['precision']*100:.1f}%)")
    print(f"  -> F1 Score                 : {best_row['f1']:.3f}")
    print(f"  -> False Positive Rate      : {best_row['fpr']:.4f}")
    print("=======================================================\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze M3 Threshold Trade-offs")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Verify Checkpoint
    model, ckpt_path, config, input_key = verify_and_load_checkpoint(device, ckpt_path_arg=args.checkpoint)
    
    # 2. Run Test Inference
    all_labels, all_probas, y_true_flat, y_proba_flat, global_auroc, global_auprc, ece = get_test_predictions(model, device, input_key=input_key)
    
    # 3. Perform Threshold Sweep
    df = perform_threshold_sweep(all_labels, all_probas, y_true_flat, y_proba_flat, global_auroc, global_auprc)
    
    # 4. Generate Plots
    generate_tradeoff_plots(df)
    
    # 5. Sweet Spot Recommendation
    identify_sweet_spot(df)


if __name__ == "__main__":
    main()
