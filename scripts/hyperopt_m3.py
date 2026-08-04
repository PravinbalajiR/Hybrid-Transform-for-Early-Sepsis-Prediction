import argparse
import copy
import optuna
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import os
from pathlib import Path
from datetime import datetime
import sys
from torch.cuda.amp import autocast, GradScaler
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.dataset import create_cached_dataloader
from models.transformer.tact_model import TACTModel  # We use TACTModel with ablation for M3
from scripts.train import evaluate_epoch, load_config
from evaluation.utility_score import compute_utility_score, find_optimal_threshold
from evaluation.metrics import full_evaluation_report
from utils.seed import set_seed

def build_m3_model(trial, config, device):
    # Base configuration for M3
    model_config = copy.deepcopy(config)
    
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 5e-5, 5e-4, log=True)
    dropout = trial.suggest_float("dropout", 0.05, 0.35)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 96, 128])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    layers = trial.suggest_int("layers", 2, 4)
    num_heads = trial.suggest_categorical("num_heads", [4, 8])
    
    # Adjust hidden_dim to be divisible by num_heads if needed
    if hidden_dim % num_heads != 0:
        num_heads = 4 # Fallback
    
    model_config["hidden_dim"] = hidden_dim
    model_config["num_heads"] = num_heads
    model_config["layers"] = layers
    model_config["dropout"] = dropout
    
    model = TACTModel(
        input_dim=102,
        d_model=hidden_dim,
        nhead=num_heads,
        num_layers=layers,
        dropout=dropout,
        ablation_mode="mask_delta" # Full M3 uses both Mask and Delta
    ).to(device)
    
    return model, lr, weight_decay, model_config

def train_epoch_optuna(model, loader, optimizer, criterion, scaler, device, input_key="triplet"):
    model.train()
    for batch in loader:
        x = batch[input_key].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)
        pad_mask = batch["padding_mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            output = model(x, padding_mask=pad_mask)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            
            valid_mask = ~pad_mask
            loss = criterion(logits[valid_mask], y[valid_mask])
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

def objective(trial, base_config, train_loader, val_loader, device, pos_weight):
    set_seed(42)
    
    model, lr, weight_decay, model_config = build_m3_model(trial, base_config, device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Warmup + ReduceLROnPlateau is complex for short optuna runs. We will just use Plateau.
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    scaler = GradScaler()
    
    epochs = 15 # Cap epochs for search speed
    best_score = -999.0
    
    for epoch in range(1, epochs + 1):
        train_epoch_optuna(model, train_loader, optimizer, criterion, scaler, device)
        val_loss, val_labels, val_probas = evaluate_epoch(model, val_loader, criterion, device, input_key="triplet")
        
        thresh, val_utility = find_optimal_threshold(val_labels, val_probas, n_thresholds=20)
        
        from sklearn.metrics import roc_auc_score, average_precision_score
        try:
            val_auprc = average_precision_score(np.concatenate(val_labels), np.concatenate(val_probas))
        except ValueError:
            val_auprc = 0.0
            
        # The user's custom combined metric
        score = (0.6 * val_utility) + (0.4 * val_auprc)
        
        scheduler.step(score)
        
        if score > best_score:
            best_score = score
            
        trial.report(score, epoch)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
            
    return best_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/tact.yaml")
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()
    
    base_config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cache_path = Path(__file__).parent.parent / "data" / "processed" / "full_dataset_cache.pt"
    if not cache_path.exists():
        outer_cache_path = Path(__file__).parent.parent.parent / "processed" / "full_dataset_cache.pt"
        if outer_cache_path.exists():
            cache_path = outer_cache_path
        else:
            raise FileNotFoundError(f"Cached dataset not found at {cache_path}")
            
    print(f"Loading cached dataset from {cache_path}...")
    cache_dict = torch.load(cache_path)
    
    train_samples, val_samples = [], []
    for pid, item in cache_dict.items():
        item["patient_id"] = pid
        if item["split"] == "train":
            train_samples.append(item)
        elif item["split"] == "val":
            val_samples.append(item)
            
    num_workers = min(os.cpu_count() or 2, 4)
    train_loader = create_cached_dataloader(
        train_samples, batch_size=base_config.get("batch_size", 32), shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    val_loader = create_cached_dataloader(
        val_samples, batch_size=base_config.get("batch_size", 32), shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    
    pos_weight = torch.tensor([47.66], device=device)
    
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5))
    
    study.optimize(lambda trial: objective(trial, base_config, train_loader, val_loader, device, pos_weight), n_trials=args.trials)
    
    print("\n==================================")
    print("  OPTUNA HYPERPARAMETER SEARCH    ")
    print("==================================")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Score (0.6 Utility + 0.4 AUPRC): {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Save best params
    with open("best_m3_params.json", "w") as f:
        json.dump(trial.params, f, indent=4)
        
if __name__ == "__main__":
    main()
