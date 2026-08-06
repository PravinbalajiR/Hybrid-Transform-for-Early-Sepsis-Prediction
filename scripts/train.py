import argparse
import sys
import os
import json
import yaml
import time
import shutil
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.dataset import create_cached_dataloader
from models.transformer.tact_model import TACTModel
from models.hybrid.hybrid_model import SepsisHybridModel
from training.losses import FocalLoss, UtilityAwareLoss
from evaluation.utility_score import compute_utility_score, find_optimal_threshold
from evaluation.metrics import full_evaluation_report, plot_reliability_diagram
from utils.seed import set_seed


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_model(config: dict, device: torch.device) -> nn.Module:
    model_type = str(config.get("model", "m2")).lower()
    hidden_dim = config.get("hidden_dim", 64)
    num_heads = config.get("num_heads", 4)
    layers = config.get("layers", 3)
    ablation_mode = config.get("ablation_mode", "none")
    
    if "m3_recovered" in model_type:
        from models.transformer.m3_recovered_model import M3RecoveredModel
        return M3RecoveredModel(
            input_dim=102,
            d_model=config.get("d_model", hidden_dim),
            nhead=num_heads,
            num_layers=layers,
            dim_feedforward=config.get("dim_feedforward", 128),
            dropout=config.get("dropout", 0.1),
        ).to(device)

    if "m2" in model_type or "plain" in model_type:
        input_dim = 34
    elif "m3" in model_type or "time_aware" in model_type or "tact" in model_type:
        input_dim = 34 * 3
    else:
        raise ValueError(f"Unknown model type: {model_type}")


    model = TACTModel(
        input_dim=input_dim,
        d_model=hidden_dim,
        nhead=num_heads,
        num_layers=layers,
        ablation_mode=ablation_mode,
    ).to(device)
    
    return model



def save_plots(metrics_history: dict, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = metrics_history["epoch"]

    def plot_metric(key, ylabel, title, filename):
        if key not in metrics_history or not metrics_history[key]:
            return
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, metrics_history[key], marker='o', label=key)
        if f"val_{key}" in metrics_history:
            plt.plot(epochs, metrics_history[f"val_{key}"], marker='o', label=f"val_{key}")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
        plt.savefig(save_dir / filename)
        plt.close()

    plot_metric("loss", "Loss", "Training and Validation Loss", "loss_curve.png")
    plot_metric("auroc", "AUROC", "Validation AUROC", "auroc_curve.png")
    plot_metric("auprc", "AUPRC", "Validation AUPRC", "auprc_curve.png")
    plot_metric("utility", "Utility Score", "Validation Utility", "utility_curve.png")


def train_epoch(model, loader, optimizer, criterion, scaler, device, input_key, verify_mode=False):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Training", leave=False)
    
    batch_idx = 0
    for batch in pbar:
        if verify_mode and batch_idx >= 100:
            print("\n[Verify Mode] Reached 100 batches. Stopping epoch early.")
            break
            
        x = batch[input_key].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)
        pad_mask = batch["padding_mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            output = model(x, padding_mask=pad_mask)
            if isinstance(output, tuple):
                logits, forecast_preds = output
            else:
                logits = output
                forecast_preds = None
                
            valid_mask = ~pad_mask
            
            if isinstance(criterion, dict):
                focal_l = criterion["focal"](logits[valid_mask], y[valid_mask])
                util_l = criterion["utility"](logits, y, valid_mask=valid_mask)
                loss = 0.7 * focal_l + 0.3 * util_l
                bce_loss = loss
            else:
                loss = criterion(logits[valid_mask], y[valid_mask])
                bce_loss = loss
            
            if forecast_preds is not None:
                # Forecasting target indices: MAP (4), Creatinine (19), Lactate (22), O2Sat (1), Resp (6)
                target_idx = [4, 19, 22, 1, 6]
                x_vals = x[:, :, target_idx] # (B, T, 5)
                target_delta = torch.zeros_like(x_vals)
                target_delta[:, :-1, :] = x_vals[:, 1:, :] - x_vals[:, :-1, :]
                
                mse_loss = F.mse_loss(forecast_preds[valid_mask], target_delta[valid_mask])
                smooth_l1 = F.smooth_l1_loss(forecast_preds[valid_mask], target_delta[valid_mask])
                forecast_loss = 0.2 * mse_loss + 0.05 * smooth_l1
                loss = bce_loss + forecast_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * batch["values"].size(0)
        
        mem_mb = torch.cuda.memory_allocated(device) / (1024 ** 2) if device.type == 'cuda' else 0
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "vram_mb": f"{mem_mb:.0f}"})

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device, input_key):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probas = []

    pbar = tqdm(loader, desc="Evaluating", leave=False)
    for batch in pbar:
        x = batch[input_key].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)
        pad_mask = batch["padding_mask"].to(device, non_blocking=True)

        with autocast():
            output = model(x, padding_mask=pad_mask)
            if isinstance(output, tuple):
                logits, forecast_preds = output
            else:
                logits = output
                forecast_preds = None
                
            valid_mask = ~pad_mask
            
            if isinstance(criterion, dict):
                focal_l = criterion["focal"](logits[valid_mask], y[valid_mask])
                util_l = criterion["utility"](logits, y, valid_mask=valid_mask)
                loss = 0.7 * focal_l + 0.3 * util_l
                bce_loss = loss
            else:
                loss = criterion(logits[valid_mask], y[valid_mask])
                bce_loss = loss
            
            if forecast_preds is not None:
                target_idx = [4, 19, 22, 1, 6]
                x_vals = x[:, :, target_idx]
                target_delta = torch.zeros_like(x_vals)
                target_delta[:, :-1, :] = x_vals[:, 1:, :] - x_vals[:, :-1, :]
                
                mse_loss = F.mse_loss(forecast_preds[valid_mask], target_delta[valid_mask])
                smooth_l1 = F.smooth_l1_loss(forecast_preds[valid_mask], target_delta[valid_mask])
                loss = bce_loss + 0.2 * mse_loss + 0.05 * smooth_l1

        total_loss += loss.item() * batch["values"].size(0)
        probas = torch.sigmoid(logits).cpu().numpy()
        labels = batch["labels"].numpy()

        for i in range(len(batch["patient_ids"])):
            length = batch["lengths"][i].item()
            all_probas.append(probas[i, :length])
            all_labels.append(labels[i, :length])

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, all_labels, all_probas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", action="store_true", help="Resume from best checkpoint if exists")
    parser.add_argument("--verify", action="store_true", help="Run a quick 100-batch verification to ensure pipeline works")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    model_name = config.get("model", "plain_transformer")
    
    # Setup Experiment Directory
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    exp_dir = Path(__file__).parent.parent / "experiments" / model_name / run_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy config to experiment dir
    shutil.copy(config_path, exp_dir / "config.yaml")

    # Set up logging & tensorboard
    tb_writer = SummaryWriter(log_dir=str(exp_dir / "logs"))
    history_csv = exp_dir / "history.csv"
    with open(history_csv, "w") as f:
        f.write("epoch,train_loss,val_loss,val_auroc,val_auprc,val_utility,lr,time_sec\n")
        
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Trainer] GPU Available: {torch.cuda.is_available()} | Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Ver: {torch.version.cuda}")
    else:
        print("  WARNING: GPU unavailable. Reconnect to a GPU runtime. Do not continue on CPU.")
        # We don't exit here so local testing works, but it's strongly discouraged.

    # 1. Load Precomputed Cached Dataset
    possible_cache_paths = [
        Path(__file__).parent.parent / "data" / "processed" / "full_dataset_cache.pt",
        Path("/content/drive/MyDrive/Sepsis-Hybrid-Transformer/processed/full_dataset_cache.pt"),
        Path("/content/drive/MyDrive/Sepsis-Hybrid-Transformer/data/processed/full_dataset_cache.pt"),
        Path(__file__).parent.parent.parent / "processed" / "full_dataset_cache.pt",
    ]
    cache_path = None
    for p in possible_cache_paths:
        if p.exists():
            cache_path = p
            break
            
    if cache_path is None:
        raise FileNotFoundError("Cached dataset not found in data/processed/ or Google Drive. Please mount Drive or copy full_dataset_cache.pt.")

            
    print(f"[Trainer] Loading cached dataset tensors in memory from {cache_path}...")
    cache_dict = torch.load(cache_path)
    
    train_samples, val_samples, test_samples = [], [], []
    train_labels = []
    
    for pid, item in cache_dict.items():
        item["patient_id"] = pid
        if item["split"] == "train":
            train_samples.append(item)
            train_labels.append(item["labels"].numpy())
        elif item["split"] == "val":
            val_samples.append(item)
        else:
            test_samples.append(item)
            
    print(f"[Trainer] Split Sizes - Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    # Setup DataLoaders
    # Colab GPU optimizations: pin_memory=True, multiple workers.
    num_workers = min(os.cpu_count() or 2, 4)
    train_loader = create_cached_dataloader(
        train_samples, batch_size=config.get("batch_size", 32), shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    val_loader = create_cached_dataloader(
        val_samples, batch_size=config.get("batch_size", 32), shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    test_loader = create_cached_dataloader(
        test_samples, batch_size=config.get("batch_size", 32), shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )

    # Calculate pos_weight dynamically
    all_train_y = np.concatenate(train_labels)
    n_pos = (all_train_y == 1).sum()
    n_neg = (all_train_y == 0).sum()
    pw_value = float(n_neg / n_pos) if n_pos > 0 else 1.0
    pos_weight = torch.tensor([pw_value]).to(device)
    print(f"  [imbalance] neg={n_neg:,}  pos={n_pos:,}  pos_weight={pw_value:.2f}")

    # 2. Build Model & Optimizer
    model = build_model(config, device)
    # BUG FIX: For M3 (tact with ablation_mode=none or mask_delta), use simple
    # BCEWithLogitsLoss. The compound focal+utility criterion is only needed for
    # the TACT-UGO variant (focal_only / tact_ugo ablation modes).
    # Using a dict criterion with UtilityAwareLoss on M3 was introducing a Python
    # loop over every batch row which hurt both speed and correctness.
    ablation_mode = config.get("ablation_mode", "none")
    if "tact_ugo" in model_name.lower() or ablation_mode in ["focal_only", "tact_ugo"]:
        criterion = {
            "focal": FocalLoss(pos_weight=pos_weight.item(), gamma=2.0, reduction="mean"),
            "utility": UtilityAwareLoss(base_pos_weight=pos_weight.item(), reduction="mean")
        }
    else:
        # Standard weighted BCE — correct for M2, M3, ablations, and hybrid
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 1e-4),
        weight_decay=config.get("weight_decay", 1e-4)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    scaler = GradScaler()

    # input_key: 'values' for M2 / plain_transformer, 'triplet' for M3, M4, TACT
    input_key = "values" if ("m2" in model_name.lower() or "plain" in model_name.lower()) else "triplet"

    
    start_epoch = 1
    best_val_utility = -np.inf
    epochs_no_improve = 0
    metrics_history = {"epoch": [], "loss": [], "val_loss": [], "val_auroc": [], "val_auprc": [], "val_utility": []}
    
    best_ckpt_path = exp_dir / "checkpoints"
    best_ckpt_path.mkdir(exist_ok=True)

    # Resume Logic
    if args.resume:
        # Search for best checkpoint in the model's experiment dir (across runs)
        parent_dir = exp_dir.parent
        checkpoints = list(parent_dir.glob("*/checkpoints/best_*.pt"))
        if checkpoints:
            latest_ckpt = max(checkpoints, key=os.path.getctime)
            print(f"[Trainer] Resuming from checkpoint: {latest_ckpt}")
            ckpt = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt["epoch"] + 1
            best_val_utility = ckpt["best_metric"]
        else:
            print("[Trainer] No checkpoint found to resume from.")

    # 3. Training Loop
    epochs = 1 if args.verify else config.get("epochs", 30)
    early_stopping_patience = config.get("patience", 10)

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, device, input_key, verify_mode=args.verify)
        val_loss, val_labels, val_probas = evaluate_epoch(model, val_loader, criterion, device, input_key)
        
        thresh, val_utility = find_optimal_threshold(val_labels, val_probas, n_thresholds=20)
        
        # Compute subset metrics manually for quick logging
        from sklearn.metrics import roc_auc_score, average_precision_score
        try:
            val_auroc = roc_auc_score(np.concatenate(val_labels), np.concatenate(val_probas))
            val_auprc = average_precision_score(np.concatenate(val_labels), np.concatenate(val_probas))
        except ValueError:
            val_auroc, val_auprc = 0.0, 0.0

        epoch_time = time.time() - t0
        lr_current = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Utility: {val_utility:.4f} | AUROC: {val_auroc:.4f} | Time: {epoch_time:.1f}s")
        
        # Log to TensorBoard
        tb_writer.add_scalar("Loss/train", train_loss, epoch)
        tb_writer.add_scalar("Loss/val", val_loss, epoch)
        tb_writer.add_scalar("Metrics/val_utility", val_utility, epoch)
        tb_writer.add_scalar("Metrics/val_auroc", val_auroc, epoch)
        tb_writer.add_scalar("Metrics/val_auprc", val_auprc, epoch)
        tb_writer.add_scalar("Hyperparameters/lr", lr_current, epoch)
        
        # Log to CSV
        with open(history_csv, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{val_auroc:.6f},{val_auprc:.6f},{val_utility:.6f},{lr_current},{epoch_time:.2f}\n")
            
        # Update History Dict
        metrics_history["epoch"].append(epoch)
        metrics_history["loss"].append(train_loss)
        metrics_history["val_loss"].append(val_loss)
        metrics_history["val_auroc"].append(val_auroc)
        metrics_history["val_auprc"].append(val_auprc)
        metrics_history["val_utility"].append(val_utility)

        scheduler.step(val_utility)

        if val_utility > best_val_utility:
            best_val_utility = val_utility
            epochs_no_improve = 0

            # Clean old best checkpoints
            for f in best_ckpt_path.glob("best_*.pt"):
                f.unlink()

            ckpt_name = f"best_{model_name}_auroc{val_auroc:.3f}_epoch{epoch:02d}.pt"
            # BUG FIX: Save both the full checkpoint dict AND a flat state_dict
            # so evaluate_robustness.py can load it with model.load_state_dict() directly.
            full_ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_metric": best_val_utility,
                "config": config
            }
            torch.save(full_ckpt, best_ckpt_path / ckpt_name)
            # Also save a clean state_dict for simple loading in downstream scripts
            torch.save(model.state_dict(), exp_dir / "best_model.pt")
            print(f"  -> Best model saved: {ckpt_name}")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement for {epochs_no_improve} epochs.")
            
        if epochs_no_improve >= early_stopping_patience:
            print(f"\n[Trainer] Early stopping triggered after {epoch} epochs!")
            break

    # Save Plots
    print("[Trainer] Generating learning curves...")
    save_plots(metrics_history, exp_dir / "plots")
    
    # 4. Final Evaluation on Test Set
    print("\n[Trainer] Evaluating Best Model on Test Set...")
    print(f"  [Ablation Mode] Active ablation: {ablation_mode}")
    # Load Best Model
    best_ckpts = list(best_ckpt_path.glob("best_*.pt"))
    if best_ckpts:
        best_ckpt = max(best_ckpts, key=os.path.getctime)
        print(f"  [Checkpoint] Loading best checkpoint: {best_ckpt}")
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
    else:
        print("  [Checkpoint] WARNING: No checkpoint found! Using current model weights.")

    test_loss, test_labels, test_probas = evaluate_epoch(model, test_loader, criterion, device, input_key)

    # Re-evaluate optimal threshold on val, apply to test
    _, val_labels, val_probas = evaluate_epoch(model, val_loader, criterion, device, input_key)
    best_thresh, val_best_u = find_optimal_threshold(val_labels, val_probas, n_thresholds=20)
    print(f"  [Threshold] Optimal threshold on Validation set: {best_thresh:.4f} (Val Utility: {val_best_u:.4f})")

    test_preds = [(p >= best_thresh).astype(int) for p in test_probas]

    test_utility = compute_utility_score(test_labels, test_preds)
    
    report = full_evaluation_report(
        test_labels, test_probas, test_preds,
        utility_score=test_utility,
        split_name=f"test_{model_name}",
    )
    
    # Save final metrics JSON
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=4)
        
    plot_reliability_diagram(
        y_true=np.concatenate(test_labels),
        y_proba=np.concatenate(test_probas),
        save_path=str(exp_dir / "plots" / "reliability_diagram.png")
    )
    
    print(f"\n=======================================================")
    print(f"  FINAL TEST RESULTS : {model_name.upper()}")
    print(f"=======================================================")
    print(f"  Utility Score : {report['utility_score']:.4f}")
    print(f"  AUROC         : {report['auroc']:.4f}")
    print(f"  AUPRC         : {report['auprc']:.4f}")
    print(f"  F1            : {report['f1']:.4f}")
    print(f"=======================================================\n")
    
    if args.verify:
        print("[Trainer] Verification Phase Complete. Forward/Backward passes and metrics are working correctly.")
        
    tb_writer.close()

if __name__ == "__main__":
    main()
