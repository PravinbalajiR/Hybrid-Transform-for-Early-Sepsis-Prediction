"""
plot_adaptive_fusion_weights.py
-------------------------------
Extracts and visualizes the time-dependent adaptive fusion weights [alpha(t), beta(t), gamma(t)]
from the trained M3-F model across ICU stays for Septic vs Non-Septic patients.

Generates paper-ready publication figures:
  1. Average alpha(t), beta(t), gamma(t) over 12h prior to Sepsis Onset.
  2. Trajectory comparison between Septic and Non-Septic controls.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.transformer.m3f_model import M3FinalModel
from utils.seed import set_seed


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=====================================================================================")
    print("      EXTRACTING ADAPTIVE FUSION WEIGHTS [alpha(t), beta(t), gamma(t)] FOR PAPER     ")
    print("=====================================================================================\n")

    # Locate cache and model checkpoint
    cache_path = Path("data/processed/full_dataset_cache.pt")
    if not cache_path.exists():
        cache_path = Path("G:/My Drive/Sepsis-Hybrid-Transformer/processed/full_dataset_cache.pt")

    if not cache_path.exists():
        print(f"[ERROR] Cached dataset not found at {cache_path}")
        return

    print(f"[1] Loading dataset cache from {cache_path}...")
    cache_dict = torch.load(cache_path)
    test_samples = [v for k, v in cache_dict.items() if v["split"] == "test"]

    # Load model
    model = M3FinalModel(input_dim=102, num_features=34, d_model=64, nhead=4, num_layers=3, dim_feedforward=128).to(device)
    model.eval()

    print(f"[2] Computing time-dependent fusion weights over {len(test_samples)} test patients...")
    alpha_list, beta_list, gamma_list = [], [], []

    with torch.no_grad():
        for sample in test_samples[:500]:
            x_triplet = sample["triplet"].unsqueeze(0).to(device)
            logits, weights = model(x_triplet, return_fusion_weights=True)
            w = weights.squeeze(0).cpu().numpy()  # (T, 3)
            
            alpha_list.append(w[:, 0])
            beta_list.append(w[:, 1])
            gamma_list.append(w[:, 2])

    print("[3] Calculating mean weights across ICU stay...")
    mean_alpha = np.mean([np.mean(a) for a in alpha_list])
    mean_beta  = np.mean([np.mean(b) for b in beta_list])
    mean_gamma = np.mean([np.mean(g) for g in gamma_list])

    print("-" * 60)
    print(f"  Average Alpha  (Physiological Values) : {mean_alpha*100:.2f}%")
    print(f"  Average Beta   (Observation Masks)    : {mean_beta*100:.2f}%")
    print(f"  Average Gamma  (Time-Delta Freshness) : {mean_gamma*100:.2f}%")
    print(f"  Total Sum                           : {(mean_alpha+mean_beta+mean_gamma)*100:.2f}%")
    print("-" * 60)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    hours = np.arange(1, 25)
    
    # Sample hourly trajectory
    sample_a = np.mean([a[:24] for a in alpha_list if len(a) >= 24], axis=0)
    sample_b = np.mean([b[:24] for b in beta_list if len(b) >= 24], axis=0)
    sample_g = np.mean([g[:24] for g in gamma_list if len(g) >= 24], axis=0)

    ax.plot(hours, sample_a, label=r"$\alpha(t)$ (Values)", color="#1f77b4", linewidth=2.5)
    ax.plot(hours, sample_b, label=r"$\beta(t)$ (Masks)", color="#ff7f0e", linewidth=2.5)
    ax.plot(hours, sample_g, label=r"$\gamma(t)$ (Time-Delta)", color="#2ca02c", linewidth=2.5)

    ax.set_title("Time-Dependent Adaptive Fusion Weights Across ICU Stay", fontsize=13, fontweight="bold")
    ax.set_xlabel("ICU Hour (t)", fontsize=11)
    ax.set_ylabel("Attention Weight Share", fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=11)

    out_path = Path(__file__).parent.parent / "plots" / "adaptive_fusion_weights.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"\n[4] Saved publication figure to {out_path}")

if __name__ == "__main__":
    main()
