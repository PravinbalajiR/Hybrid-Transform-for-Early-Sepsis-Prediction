"""
transformer_baseline.py
------------------------
Training and evaluation script for Transformer Baseline Models:
  - Model M2: Plain Transformer (naive mean imputation)
  - Model M3: Time-Aware Transformer (values + mask + time-delta triplet)

Evaluates using the official PhysioNet 2019 Utility Score and threshold optimization.

Usage:
  python baselines/transformer_baseline.py --mode plain
  python baselines/transformer_baseline.py --mode time_aware
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocessing.load_data import load_dataset, ALL_FEATURE_COLS, LABEL_COL, DEFAULT_SET_A, DEFAULT_SET_B
from preprocessing.split import make_splits, save_splits, load_splits
from preprocessing.normalize import Normalizer
from preprocessing.dataset import create_dataloader, SepsisDataset
from models.transformer.transformer_encoder import SepsisTransformer
from evaluation.utility_score import compute_utility_score, find_optimal_threshold
from evaluation.metrics import compute_classification_metrics, full_evaluation_report
from utils.seed import set_seed
from utils.logger import ExperimentLogger


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    input_key: str = "triplet",
) -> float:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        x = batch[input_key].to(device)              # (B, T, F or 3F)
        y = batch["labels"].to(device)               # (B, T)
        pad_mask = batch["padding_mask"].to(device)  # (B, T) True for padded

        optimizer.zero_grad()
        logits = model(x, padding_mask=pad_mask)     # (B, T)

        # Mask out padded positions from loss computation
        valid_mask = ~pad_mask
        loss = criterion(logits[valid_mask], y[valid_mask])

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch["values"].size(0)

    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    input_key: str = "triplet",
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Get predicted probabilities and ground truth labels per patient."""
    model.eval()
    all_labels = []
    all_probas = []

    for batch in dataloader:
        x = batch[input_key].to(device)
        pad_mask = batch["padding_mask"].to(device)

        logits = model(x, padding_mask=pad_mask)     # (B, T)
        probas = torch.sigmoid(logits).cpu().numpy()
        labels = batch["labels"].numpy()

        for i in range(len(batch["patient_ids"])):
            length = batch["lengths"][i].item()
            all_probas.append(probas[i, :length])
            all_labels.append(labels[i, :length])

    return all_labels, all_probas


def run_experiment(
    mode: str = "time_aware",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-4,
    d_model: int = 64,
    seed: int = 42,
    max_patients: Optional[int] = None,
) -> dict:
    """Run full training and evaluation pipeline."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Baseline Run] Mode: {mode} | Device: {device} | LR: {lr} | Epochs: {epochs}")

    # 1. Load dataset
    patient_dfs, source_map = load_dataset(
        set_a_dir=DEFAULT_SET_A,
        set_b_dir=DEFAULT_SET_B,
        max_patients=max_patients,
        verbose=True,
    )

    # 2. Split (Set A -> Train/Val, Set B -> Test)
    splits_dir = Path(__file__).parent.parent / "data" / "splits"
    patient_records = [
        {
            "PatientID": df["PatientID"].iloc[0],
            "Source": df["Source"].iloc[0],
            "SepsisLabel": int(df[LABEL_COL].max()),
        }
        for df in patient_dfs
    ]
    patient_df = pd.DataFrame(patient_records)

    if (splits_dir / "train_ids.json").exists():
        train_ids, val_ids, test_ids = load_splits(splits_dir)
    else:
        train_ids, val_ids, test_ids = make_splits(patient_df)
        save_splits(train_ids, val_ids, test_ids, splits_dir)

    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)

    dfs_train = [df for df in patient_dfs if df["PatientID"].iloc[0] in train_set]
    dfs_val   = [df for df in patient_dfs if df["PatientID"].iloc[0] in val_set]
    dfs_test  = [df for df in patient_dfs if df["PatientID"].iloc[0] in test_set]

    print(f"Split sizes: Train={len(dfs_train)}, Val={len(dfs_val)}, Test={len(dfs_test)}")

    # 3. Z-score Normalize (Fit on train split only!)
    norm = Normalizer()
    norm.fit(dfs_train, ALL_FEATURE_COLS, verbose=False)
    dfs_train = norm.transform(dfs_train)
    dfs_val   = norm.transform(dfs_val)
    dfs_test  = norm.transform(dfs_test)

    # 4. DataLoaders
    train_loader = create_dataloader(dfs_train, batch_size=batch_size, shuffle=True)
    val_loader   = create_dataloader(dfs_val,   batch_size=batch_size, shuffle=False)
    test_loader  = create_dataloader(dfs_test,  batch_size=batch_size, shuffle=False)

    # 5. Initialize Model
    input_key = "triplet" if mode == "time_aware" else "values"
    input_dim = 34 * 3 if mode == "time_aware" else 34

    model = SepsisTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=4,
        num_layers=3,
    ).to(device)

    # Pos weight for class imbalance — computed dynamically from training labels
    all_train_labels = np.concatenate([
        df[LABEL_COL].values for df in dfs_train
    ])
    n_neg = (all_train_labels == 0).sum()
    n_pos = (all_train_labels == 1).sum()
    pw_value = float(n_neg / n_pos) if n_pos > 0 else 1.0
    print(f"  [imbalance] neg={n_neg:,}  pos={n_pos:,}  pos_weight={pw_value:.2f}")
    pos_weight = torch.tensor([pw_value]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    logger = ExperimentLogger(
        log_dir=Path(__file__).parent.parent / "experiments" / "logs",
        experiment_name=f"transformer_{mode}",
    )

    # 6. Training Loop
    best_val_utility = -np.inf
    best_thresh = 0.5
    early_stopping_patience = 5
    epochs_no_improve = 0
    best_model_path = Path(__file__).parent.parent / "experiments" / "checkpoints" / f"best_transformer_{mode}.pt"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device, input_key=input_key)

        val_labels, val_probas = evaluate_model(model, val_loader, device, input_key=input_key)
        thresh, val_utility = find_optimal_threshold(val_labels, val_probas, n_thresholds=20)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {loss:.4f} | Val Utility: {val_utility:.4f} (thresh={thresh:.2f})")
        logger.log(step=epoch, metrics={"loss": loss, "val_utility": val_utility, "threshold": thresh})

        scheduler.step(val_utility)

        if val_utility > best_val_utility:
            best_val_utility = val_utility
            best_thresh = thresh
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Best model saved! (Val Utility: {best_val_utility:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement for {epochs_no_improve} epochs.")
            
        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch} epochs!")
            break

    # 7. Evaluate on Held-out Test Set (Set B)
    print("\n[Evaluating on Test Set B...]")
    # Load best model weights before evaluation
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path))
        print("  -> Loaded best model weights for test evaluation.")
        
    test_labels, test_probas = evaluate_model(model, test_loader, device, input_key=input_key)
    test_preds = [(p >= best_thresh).astype(int) for p in test_probas]
    test_utility = compute_utility_score(test_labels, test_preds)

    report = full_evaluation_report(
        test_labels, test_probas, test_preds,
        utility_score=test_utility,
        split_name=f"test_set_b_{mode}",
    )
    logger.summary(report)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["plain", "time_aware"], default="time_aware")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_patients", type=int, default=None)
    args = parser.parse_args()

    run_experiment(
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_patients=args.max_patients,
    )
