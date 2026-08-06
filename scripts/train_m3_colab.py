"""
train_m3_colab.py
-----------------
Self-contained Google Colab training script for the M3 ablation study.

Trains all four variants in sequence:
  1. M2  (baseline)              — input: values only          (34-dim)
  2. M3  (full triplet)          — input: values+mask+delta   (102-dim, ablation=none)
  3. M3-mask (mask only ablation)— input: values+mask+zeros   (102-dim, ablation=mask_only)
  4. M3-delta (delta only ablation)— input: values+zeros+delta (102-dim, ablation=delta_only)

At the end, prints a clean comparison table for the paper.

Usage (Colab cell):
  !python scripts/train_m3_colab.py

Requires:
  - Google Drive mounted at /content/drive
  - full_dataset_cache.pt in Drive at:
    /content/drive/MyDrive/Sepsis-Hybrid-Transformer/processed/full_dataset_cache.pt
  - Code repo cloned to /content/code
"""

from __future__ import annotations

import sys
import os
import json
import time
import math
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.dataset import create_cached_dataloader
from models.transformer.tact_model import TACTModel
from evaluation.utility_score import compute_utility_score, find_optimal_threshold
from utils.seed import set_seed


# -----------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------

DRIVE_CACHE_PATH = Path(
    "/content/drive/MyDrive/Sepsis-Hybrid-Transformer/processed/full_dataset_cache.pt"
)
DRIVE_SAVE_ROOT = Path(
    "/content/drive/MyDrive/Sepsis-Hybrid-Transformer/experiments/m3_ablation"
)

SHARED_HPARAMS = dict(
    d_model=64,
    nhead=4,
    num_layers=3,
    dim_feedforward=128,
    dropout=0.1,
    max_len=500,
    epochs=30,
    batch_size=32,
    lr=1e-4,
    weight_decay=1e-4,
    patience=10,
    seed=42,
)

VARIANTS = [
    {
        "name": "M2",
        "input_key": "values",
        "input_dim": 34,
        "ablation_mode": "none",
        "description": "Baseline: imputed values only",
    },
    {
        "name": "M3",
        "input_key": "triplet",
        "input_dim": 102,
        "ablation_mode": "none",
        "description": "Full triplet: values + mask + log1p(delta)",
    },
    {
        "name": "M3-mask",
        "input_key": "triplet",
        "input_dim": 102,
        "ablation_mode": "mask_only",
        "description": "Ablation: values + mask only (delta zeroed)",
    },
    {
        "name": "M3-delta",
        "input_key": "triplet",
        "input_dim": 102,
        "ablation_mode": "delta_only",
        "description": "Ablation: values + log1p(delta) only (mask zeroed)",
    },
]


# -----------------------------------------------------------------
# Sanity check: forward pass for every variant
# -----------------------------------------------------------------

def run_sanity_check():
    print("\n" + "=" * 60)
    print("  SANITY CHECK: Forward pass for all 4 variants")
    print("=" * 60)

    B, T = 4, 48
    device = torch.device("cpu")

    for v in VARIANTS:
        model = TACTModel(
            input_dim=v["input_dim"],
            d_model=SHARED_HPARAMS["d_model"],
            nhead=SHARED_HPARAMS["nhead"],
            num_layers=SHARED_HPARAMS["num_layers"],
            dim_feedforward=SHARED_HPARAMS["dim_feedforward"],
            dropout=0.0,
            max_len=SHARED_HPARAMS["max_len"],
            ablation_mode=v["ablation_mode"],
        ).to(device)

        x = torch.randn(B, T, v["input_dim"])
        pad_mask = torch.zeros(B, T, dtype=torch.bool)
        pad_mask[:, 40:] = True

        out = model(x, padding_mask=pad_mask)

        assert out.shape == (B, T), (
            f"[{v['name']}] Expected ({B},{T}), got {out.shape}"
        )

        loss = out[~pad_mask].mean()
        loss.backward()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad), (
            f"[{v['name']}] Some parameters have no gradient!"
        )

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  [{v['name']:10s}]  output={out.shape}  params={n_params:,}  OK")

    print("  All sanity checks passed.\n")


# -----------------------------------------------------------------
# Training helpers
# -----------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, input_key):
    model.train()
    total_loss, n = 0.0, 0

    for batch in tqdm(loader, desc="  train", leave=False):
        x        = batch[input_key].to(device, non_blocking=True)
        y        = batch["labels"].to(device, non_blocking=True)
        pad_mask = batch["padding_mask"].to(device, non_blocking=True)
        valid    = ~pad_mask

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(x, padding_mask=pad_mask)
            loss   = criterion(logits[valid], y[valid])

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)
        n += x.size(0)

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_one_epoch(model, loader, criterion, device, input_key):
    model.eval()
    total_loss, n = 0.0, 0
    all_labels, all_probas = [], []

    for batch in tqdm(loader, desc="  eval ", leave=False):
        x        = batch[input_key].to(device, non_blocking=True)
        y        = batch["labels"].to(device, non_blocking=True)
        pad_mask = batch["padding_mask"].to(device, non_blocking=True)
        valid    = ~pad_mask

        with autocast():
            logits = model(x, padding_mask=pad_mask)
            loss   = criterion(logits[valid], y[valid])

        total_loss += loss.item() * x.size(0)
        n += x.size(0)

        probas  = torch.sigmoid(logits).cpu().numpy()
        labels  = batch["labels"].numpy()
        lengths = batch["lengths"]

        for i in range(len(batch["patient_ids"])):
            t = lengths[i].item()
            all_probas.append(probas[i, :t])
            all_labels.append(labels[i, :t])

    return total_loss / max(n, 1), all_labels, all_probas


# -----------------------------------------------------------------
# Single variant: train + evaluate
# -----------------------------------------------------------------

def train_variant(variant, train_loader, val_loader, test_loader,
                  pos_weight, device, save_dir):
    hp   = SHARED_HPARAMS
    name = variant["name"]

    print(f"\n{'=' * 60}")
    print(f"  Training: {name}  —  {variant['description']}")
    print(f"{'=' * 60}")

    set_seed(hp["seed"])

    model = TACTModel(
        input_dim      = variant["input_dim"],
        d_model        = hp["d_model"],
        nhead          = hp["nhead"],
        num_layers     = hp["num_layers"],
        dim_feedforward= hp["dim_feedforward"],
        dropout        = hp["dropout"],
        max_len        = hp["max_len"],
        ablation_mode  = variant["ablation_mode"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight]).to(device)
    )
    optimizer = AdamW(model.parameters(), lr=hp["lr"],
                      weight_decay=hp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    scaler    = GradScaler()

    best_utility = -np.inf
    best_state   = None
    no_improve   = 0
    input_key    = variant["input_key"]

    for epoch in range(1, hp["epochs"] + 1):
        t0 = time.time()

        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                  scaler, device, input_key)

        _, val_labels, val_probas = evaluate_one_epoch(model, val_loader,
                                                       criterion, device, input_key)
        thresh, val_utility = find_optimal_threshold(val_labels, val_probas,
                                                     n_thresholds=50)
        try:
            val_auroc = roc_auc_score(np.concatenate(val_labels),
                                      np.concatenate(val_probas))
        except ValueError:
            val_auroc = 0.0

        scheduler.step(val_utility)
        elapsed = time.time() - t0

        print(
            f"  Ep {epoch:02d}/{hp['epochs']} | loss={tr_loss:.4f} | "
            f"val_utility={val_utility:.4f} | val_auroc={val_auroc:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_utility > best_utility:
            best_utility = val_utility
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
            print(f"    -> New best  (utility={val_utility:.4f})")
        else:
            no_improve += 1
            if no_improve >= hp["patience"]:
                print(f"    -> Early stop after {epoch} epochs.")
                break

    # ── Test evaluation ──────────────────────────────────────────────
    print(f"\n  Final evaluation on test set ({name})...")
    model.load_state_dict(best_state)

    # Derive threshold on val
    _, val_labels, val_probas = evaluate_one_epoch(model, val_loader,
                                                   criterion, device, input_key)
    best_thresh, _ = find_optimal_threshold(val_labels, val_probas, n_thresholds=100)

    _, test_labels, test_probas = evaluate_one_epoch(model, test_loader,
                                                     criterion, device, input_key)
    test_preds = [(p >= best_thresh).astype(int) for p in test_probas]

    flat_y  = np.concatenate(test_labels)
    flat_p  = np.concatenate(test_probas)
    flat_yh = np.concatenate(test_preds)

    try:
        auroc = roc_auc_score(flat_y, flat_p)
        auprc = average_precision_score(flat_y, flat_p)
        f1    = f1_score(flat_y, flat_yh)
    except ValueError:
        auroc = auprc = f1 = 0.0

    utility = compute_utility_score(test_labels, test_preds)

    results = dict(
        name          = name,
        description   = variant["description"],
        ablation_mode = variant["ablation_mode"],
        input_dim     = variant["input_dim"],
        auroc         = round(auroc, 4),
        auprc         = round(auprc, 4),
        f1            = round(f1, 4),
        utility       = round(utility, 4),
        threshold     = round(best_thresh, 4),
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    key = name.lower().replace("-", "_")
    torch.save(best_state, save_dir / f"{key}_best.pt")
    with open(save_dir / f"{key}_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"  {name}: AUROC={auroc:.4f}  AUPRC={auprc:.4f}  "
          f"F1={f1:.4f}  Utility={utility:.4f}")

    return results


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n[M3 Ablation Study]")
    print(f"  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")

    run_sanity_check()

    # Load cache
    cache_candidates = [
        DRIVE_CACHE_PATH,
        Path(__file__).parent.parent / "data" / "processed" / "full_dataset_cache.pt",
    ]
    cache_path = next((p for p in cache_candidates if p.exists()), None)
    if cache_path is None:
        raise FileNotFoundError(
            "full_dataset_cache.pt not found. Mount Google Drive or check path.\n"
            f"Tried:\n" + "\n".join(f"  {p}" for p in cache_candidates)
        )

    print(f"\n[Data] Loading cache: {cache_path}")
    cache = torch.load(cache_path)

    train_samples, val_samples, test_samples = [], [], []
    train_labels_raw = []

    for pid, item in cache.items():
        item["patient_id"] = pid
        split = item["split"]
        if split == "train":
            train_samples.append(item)
            train_labels_raw.append(item["labels"].numpy())
        elif split == "val":
            val_samples.append(item)
        else:
            test_samples.append(item)

    print(f"  Train={len(train_samples):,}  Val={len(val_samples):,}  "
          f"Test={len(test_samples):,}")

    all_y = np.concatenate(train_labels_raw)
    n_pos = (all_y == 1).sum()
    n_neg = (all_y == 0).sum()
    pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0
    print(f"  pos_weight={pos_weight:.2f}  (neg={n_neg:,}  pos={n_pos:,})")

    nw  = min(os.cpu_count() or 2, 4)
    bs  = SHARED_HPARAMS["batch_size"]
    pin = device.type == "cuda"

    train_loader = create_cached_dataloader(
        train_samples, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0),
    )
    val_loader = create_cached_dataloader(
        val_samples, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0),
    )
    test_loader = create_cached_dataloader(
        test_samples, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0),
    )

    run_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = DRIVE_SAVE_ROOT / f"run_{run_ts}"

    all_results = []
    for variant in VARIANTS:
        result = train_variant(
            variant, train_loader, val_loader, test_loader,
            pos_weight=pos_weight, device=device, save_dir=save_dir,
        )
        all_results.append(result)

    # Final table
    print("\n\n" + "=" * 78)
    print("  M3 ABLATION STUDY  —  FINAL RESULTS")
    print("=" * 78)
    hdr = f"  {'Model':<12} {'Dim':>4} {'AUROC':>7} {'AUPRC':>7} {'F1':>7} {'Utility':>8}"
    print(hdr)
    print("-" * 78)
    for r in all_results:
        print(
            f"  {r['name']:<12} {r['input_dim']:>4} "
            f"{r['auroc']:>7.4f} {r['auprc']:>7.4f} "
            f"{r['f1']:>7.4f} {r['utility']:>8.4f}  {r['description']}"
        )
    print("=" * 78)

    with open(save_dir / "ablation_summary.json", "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\n  Saved to: {save_dir}")


if __name__ == "__main__":
    main()
