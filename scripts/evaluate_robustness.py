import argparse
import json
import os
from pathlib import Path
import copy
import numpy as np
import torch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.dataset import create_cached_dataloader
from preprocessing.load_data import LAB_COLS, ALL_FEATURE_COLS
from models.transformer.tact_model import TACTModel
from scripts.train import evaluate_epoch, load_config
from evaluation.utility_score import find_optimal_threshold

def inject_clinical_missingness(cache_dict, drop_prob=0.0):
    """
    Simulates clinical lab delays.
    Iterates over all test patients and randomly drops 'drop_prob' % 
    of existing laboratory measurements while keeping vital signs perfectly intact.
    """
    if drop_prob == 0.0:
        return cache_dict
        
    corrupted_cache = {}
    
    # Identify which feature indices belong to LAB_COLS
    lab_indices = [ALL_FEATURE_COLS.index(c) for c in LAB_COLS if c in ALL_FEATURE_COLS]
    
    for pid, item in cache_dict.items():
        if item["split"] != "test":
            continue
            
        corrupted_item = copy.deepcopy(item)
        values = corrupted_item["values"].numpy()
        mask = corrupted_item["mask"].numpy()
        time_delta = corrupted_item["time_delta"].numpy()
        
        T, F = mask.shape
        
        # For every timestep, for every lab feature, if it was observed (mask=1), 
        # drop it with drop_prob probability.
        for t in range(T):
            for f in lab_indices:
                if mask[t, f] == 1.0:
                    if np.random.rand() < drop_prob:
                        # Drop this observation
                        mask[t, f] = 0.0
                        values[t, f] = 0.0 # Revert to imputed mean (0.0 for z-scored data)
                        
        # Now we must forcefully RECALCULATE the time_deltas!
        # Because we dropped observations, the time since the last observation increases.
        for f in lab_indices:
            current_delta = 0.0 # assuming starting at 0
            for t in range(T):
                if mask[t, f] == 1.0:
                    current_delta = 0.0 # reset delta
                else:
                    current_delta += 1.0 # time moves forward 1 hour
                time_delta[t, f] = current_delta
                
        corrupted_item["mask"] = torch.tensor(mask, dtype=torch.float32)
        corrupted_item["values"] = torch.tensor(values, dtype=torch.float32)
        corrupted_item["time_delta"] = torch.tensor(time_delta, dtype=torch.float32)
        
        # Re-encode triplet
        triplet = np.concatenate([values * mask, mask, time_delta], axis=-1)
        corrupted_item["triplet"] = torch.tensor(triplet, dtype=torch.float32)
        
        corrupted_cache[pid] = corrupted_item
        
    return corrupted_cache

def load_best_model(config_path, ablation_mode, weights_path, device):
    config = load_config(config_path)
    input_dim = 34 if ablation_mode == "none" else 102
    
    model = TACTModel(
        input_dim=input_dim,
        d_model=config.get("hidden_dim", 64),
        nhead=config.get("num_heads", 4),
        num_layers=config.get("layers", 3),
        dropout=config.get("dropout", 0.1),
        ablation_mode=ablation_mode
    ).to(device)
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"WARNING: Weights {weights_path} not found! Will evaluate untrained model.")
        
    model.eval()
    return model

def evaluate_on_cache(model, test_cache, input_key, device, pos_weight):
    test_samples = list(test_cache.values())
    for pid, item in test_cache.items():
        item["patient_id"] = pid
        
    loader = create_cached_dataloader(test_samples, batch_size=64, shuffle=False)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    loss, labels, probas = evaluate_epoch(model, loader, criterion, device, input_key=input_key)
    thresh, utility = find_optimal_threshold(labels, probas)
    
    from sklearn.metrics import roc_auc_score, average_precision_score
    auroc = roc_auc_score(np.concatenate(labels), np.concatenate(probas))
    auprc = average_precision_score(np.concatenate(labels), np.concatenate(probas))
    
    return utility, auroc, auprc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_path = Path(__file__).parent.parent / "data" / "processed" / "full_dataset_cache.pt"
    if not cache_path.exists():
        cache_path = Path(__file__).parent.parent.parent / "processed" / "full_dataset_cache.pt"
        
    print(f"Loading original dataset from {cache_path}...")
    full_cache = torch.load(cache_path)
    
    # We need the pos_weight. Just hardcode for evaluation (doesn't affect AUROC/AUPRC, just val_loss)
    pos_weight = torch.tensor([47.66], device=device)
    
    # Find M2 (Plain) and M3 (TACT) weights
    # Note: User must provide these or they are hardcoded
    # We will assume latest run of tact for M3, and latest plain_transformer for M2
    import glob
    def get_latest_weights(prefix):
        dirs = glob.glob(f"experiments/{prefix}/run_*")
        if not dirs:
            return ""
        latest = max(dirs, key=os.path.getctime)
        return os.path.join(latest, "best_model.pt")
        
    m2_weights = get_latest_weights("plain_transformer")
    m3_weights = get_latest_weights("tact")
    
    print(f"M2 Weights: {m2_weights}")
    print(f"M3 Weights: {m3_weights}")
    
    # We will load both models. (We assume configs are default)
    m2_model = load_best_model("configs/plain_transformer.yaml", "none", m2_weights, device)
    m3_model = load_best_model("configs/tact.yaml", "mask_delta", m3_weights, device)
    
    drop_levels = [0.0, 0.25, 0.50, 0.75, 0.90]
    
    results = {"M2": [], "M3": []}
    
    for drop in drop_levels:
        print(f"\n======================================")
        print(f" EVALUATING DROP PROB: {drop*100}% LABS")
        print(f"======================================")
        
        corrupted_test_cache = inject_clinical_missingness(full_cache, drop_prob=drop)
        
        print("Evaluating M2 (Plain Transformer)...")
        u2, roc2, prc2 = evaluate_on_cache(m2_model, corrupted_test_cache, "values", device, pos_weight)
        
        print("Evaluating M3 (Time-Aware Transformer)...")
        u3, roc3, prc3 = evaluate_on_cache(m3_model, corrupted_test_cache, "triplet", device, pos_weight)
        
        print(f"M2 -> AUROC: {roc2:.4f} | AUPRC: {prc2:.4f} | Utility: {u2:.4f}")
        print(f"M3 -> AUROC: {roc3:.4f} | AUPRC: {prc3:.4f} | Utility: {u3:.4f}")
        
        results["M2"].append({"drop": drop, "auroc": roc2, "auprc": prc2, "utility": u2})
        results["M3"].append({"drop": drop, "auroc": roc3, "auprc": prc3, "utility": u3})
        
    # Save results
    with open("clinical_robustness_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n[Done] Results saved to clinical_robustness_results.json")

if __name__ == "__main__":
    main()
