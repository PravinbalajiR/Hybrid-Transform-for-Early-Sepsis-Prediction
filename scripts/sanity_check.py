import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.transformer.transformer_encoder import SepsisTransformer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Cache
    cache_path = Path(__file__).parent.parent.parent / "processed" / "full_dataset_cache.pt"
    if not cache_path.exists():
        cache_path = Path(__file__).parent.parent / "data" / "processed" / "full_dataset_cache.pt"
        
    print(f"Loading data from {cache_path}...")
    cache = torch.load(cache_path)
    
    # 2. Verify Data Leakage (Patient IDs overlap check)
    train_pids, val_pids, test_pids = set(), set(), set()
    for pid, data in cache.items():
        if data['split'] == 'train': train_pids.add(pid)
        elif data['split'] == 'val': val_pids.add(pid)
        elif data['split'] == 'test': test_pids.add(pid)
        
    print("\n--- DATA LEAKAGE CHECK ---")
    print(f"Train patients: {len(train_pids)}")
    print(f"Val patients: {len(val_pids)}")
    print(f"Test patients: {len(test_pids)}")
    assert len(train_pids.intersection(val_pids)) == 0, "LEAKAGE: Train and Val overlap!"
    assert len(train_pids.intersection(test_pids)) == 0, "LEAKAGE: Train and Test overlap!"
    assert len(val_pids.intersection(test_pids)) == 0, "LEAKAGE: Val and Test overlap!"
    print("SUCCESS: 0 patient overlap across splits. Strict patient-level separation confirmed.")

    # 3. Load M2 Model
    exp_dir = Path(__file__).parent.parent / "experiments" / "plain_transformer"
    if not exp_dir.exists():
        print(f"\nCould not find M2 experiment directory at {exp_dir}. Cannot inspect predictions.")
        return
        
    runs = list(exp_dir.glob("run_*"))
    if not runs:
        print("No runs found for M2.")
        return
    
    latest_run = max(runs, key=lambda x: x.stat().st_mtime)
    best_ckpt = list((latest_run / "checkpoints").glob("best_*.pt"))[0]
    
    print(f"\nLoading Model Checkpoint: {best_ckpt.name}")
    model = SepsisTransformer(input_dim=34, d_model=64, nhead=4, num_layers=3).to(device)
    model.load_state_dict(torch.load(best_ckpt, map_location=device)["model"])
    model.eval()

    # 4. Inspect Predictions manually
    print("\n--- MANUAL PREDICITON INSPECTION ---")
    septic_patients = [pid for pid, data in cache.items() if data['split'] == 'test' and data['labels'].sum() > 0]
    non_septic_patients = [pid for pid, data in cache.items() if data['split'] == 'test' and data['labels'].sum() == 0]
    
    inspect_pids = septic_patients[:2] + non_septic_patients[:2]
    
    for pid in inspect_pids:
        x = cache[pid]['values'].unsqueeze(0).to(device)
        y = cache[pid]['labels'].numpy()
        
        with torch.no_grad():
            logits = model(x, padding_mask=None)
            probas = torch.sigmoid(logits).cpu().numpy()[0]
            
        print(f"\nPatient {pid} (Length: {len(y)} hours)")
        print(f"True Labels : {y.astype(int)}")
        print(f"Predictions : {np.round(probas, 2)}")
        
if __name__ == "__main__":
    main()
