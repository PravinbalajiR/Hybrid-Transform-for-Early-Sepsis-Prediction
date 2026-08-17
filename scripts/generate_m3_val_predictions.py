"""
generate_m3_val_predictions.py
------------------------------
Generates and caches raw validation prediction NPZ file (m3_final_val_predictions.npz)
for M3 model (experiments/final_m3_frozen/best_m3_frozen.pt) on Validation cohort (N=2,034).
Allows zero-leakage, reproducible temporal alert policy optimization on Validation data.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from preprocessing.dataset import create_cached_dataloader
from models.transformer.tact_model import TACTModel

RESULTS_DIR = BASE_DIR / "results"
FROZEN_CKPT_PATH = BASE_DIR / "experiments" / "final_m3_frozen" / "best_m3_frozen.pt"

def main():
    print("=" * 80)
    print("   GENERATING CACHED M3 VALIDATION PREDICTIONS (N=2,034)")
    print("=" * 80)

    val_npz_path = RESULTS_DIR / "m3_final_val_predictions.npz"
    if val_npz_path.exists():
        print(f"Validation NPZ already exists: {val_npz_path}")
        data = np.load(val_npz_path, allow_pickle=True)
        print(f"  Loaded {len(data['patient_lengths']):,} validation patients ({len(data['y_true_flat']):,} hourly records).")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load frozen model
    checkpoint = torch.load(FROZEN_CKPT_PATH, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)

    model = TACTModel(
        input_dim=102,
        d_model=64,
        nhead=4,
        num_layers=3,
        dropout=0.1,
        ablation_mode="none"
    ).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Load dataset cache
    cache_path = BASE_DIR / "data" / "processed" / "full_dataset_cache.pt"
    cache_dict = torch.load(cache_path)

    val_samples = []
    for pid, v in cache_dict.items():
        v["patient_id"] = pid
        if v.get("split") == "val":
            val_samples.append(v)

    print(f"Loaded {len(val_samples):,} validation patient samples from cache.")

    val_loader = create_cached_dataloader(val_samples, batch_size=64, shuffle=False)

    val_lbls = []
    val_probs = []
    val_pids = []
    val_lengths = []

    print("Running inference on Validation split...")
    with torch.no_grad():
        for b in val_loader:
            x = b["triplet"].to(device)
            pm = b["padding_mask"].to(device)
            out = model(x, padding_mask=pm)
            logits = out[0] if isinstance(out, tuple) else out
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = b["labels"].numpy()

            for i in range(len(b["patient_ids"])):
                l = b["lengths"][i].item()
                val_probs.append(probs[i, :l])
                val_lbls.append(labels[i, :l])
                val_pids.append(b["patient_ids"][i])
                val_lengths.append(l)

    y_true_flat = np.concatenate(val_lbls)
    y_proba_flat = np.concatenate(val_probs)

    onset_hours = []
    for lbl in val_lbls:
        if lbl.max() == 1:
            onset_hours.append(int(np.argmax(lbl)))
        else:
            onset_hours.append(-1)

    np.savez_compressed(
        val_npz_path,
        patient_ids=np.array(val_pids, dtype=object),
        y_true_flat=y_true_flat,
        y_proba_flat=y_proba_flat,
        onset_hours=np.array(onset_hours),
        patient_lengths=np.array(val_lengths)
    )

    print(f"\nSaved validation prediction NPZ: {val_npz_path}")
    print(f"Total Validation Patients : {len(val_pids):,}")
    print(f"Total Hourly Records     : {len(y_true_flat):,}")
    print("=" * 80)

if __name__ == "__main__":
    main()
