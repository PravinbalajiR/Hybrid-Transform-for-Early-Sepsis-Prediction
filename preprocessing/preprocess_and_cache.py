"""
preprocess_and_cache.py
-----------------------
Preprocesses and caches the full 40,336 patient PhysioNet 2019 dataset into
compressed PyTorch tensors (`data/processed/full_dataset_cache.pt`).

Benefits:
  - Dataset loading time drops from ~15 minutes to < 5 seconds for all model runs.
  - Guarantees identical z-score normalization fit on Train split.
  - Ensures exact hospital-stratified splits (Set A -> Train/Val, Set B -> Test).

Usage:
  python preprocessing/preprocess_and_cache.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocessing.load_data import load_dataset, ALL_FEATURE_COLS, LABEL_COL, DEFAULT_SET_A, DEFAULT_SET_B
from preprocessing.split import make_splits, save_splits
from preprocessing.normalize import Normalizer
from preprocessing.masks_and_deltas import compute_masks_and_deltas_fast, encode_triplet


PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
SPLITS_DIR    = Path(__file__).parent.parent / "data" / "splits"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)


def build_and_cache_dataset(max_len: int = 168) -> None:
    start_time = time.time()
    print("\n" + "=" * 65)
    print("  PREPROCESSING & CACHING FULL PHYSIONET 2019 DATASET")
    print("=" * 65)

    # 1. Load raw PSV files
    patient_dfs, source_map = load_dataset(
        set_a_dir=DEFAULT_SET_A,
        set_b_dir=DEFAULT_SET_B,
        verbose=True,
    )

    # 2. Extract patient metadata & build splits
    print("\n[cache] Building hospital-stratified train/val/test splits...")
    records = [
        {
            "PatientID": df["PatientID"].iloc[0],
            "Source": df["Source"].iloc[0],
            "SepsisLabel": int(df[LABEL_COL].max()),
        }
        for df in patient_dfs
    ]
    patient_df = pd.DataFrame(records)
    train_ids, val_ids, test_ids = make_splits(patient_df)
    save_splits(train_ids, val_ids, test_ids, SPLITS_DIR)

    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)

    # 3. Fit Normalizer on Train split only
    print("\n[cache] Fitting Z-score Normalizer on Train split...")
    dfs_train = [df for df in patient_dfs if df["PatientID"].iloc[0] in train_set]
    norm = Normalizer()
    norm.fit(dfs_train, ALL_FEATURE_COLS, verbose=True)
    norm.save(PROCESSED_DIR / "normalizer.pkl")

    # 4. Normalize all patient DataFrames
    print("\n[cache] Applying normalization across all patients...")
    patient_dfs = norm.transform(patient_dfs, inplace=True)

    # 5. Extract triplets, masks, deltas & convert to PyTorch tensors
    print("\n[cache] Computing missingness masks & time-deltas...")
    cache_dict = {}

    for df in tqdm(patient_dfs, desc="Caching patient tensors"):
        pid = df["PatientID"].iloc[0]
        source = df["Source"].iloc[0]

        df_curr = df.iloc[:max_len] if len(df) > max_len else df
        X, masks, deltas = compute_masks_and_deltas_fast(df_curr, ALL_FEATURE_COLS)
        labels = df_curr[LABEL_COL].values.astype(np.float32)
        triplet = encode_triplet(X, masks, deltas, impute_value=0.0)

        # Determine split
        if pid in train_set:
            split_tag = "train"
        elif pid in val_set:
            split_tag = "val"
        else:
            split_tag = "test"

        cache_dict[pid] = {
            "values":     torch.tensor(np.where(masks == 1.0, X, 0.0), dtype=torch.float32),
            "mask":       torch.tensor(masks,                          dtype=torch.float32),
            "time_delta": torch.tensor(deltas,                         dtype=torch.float32),
            "triplet":    torch.tensor(triplet,                        dtype=torch.float32),
            "labels":     torch.tensor(labels,                         dtype=torch.float32),
            "length":     torch.tensor(len(df_curr),                   dtype=torch.long),
            "source":     source,
            "split":      split_tag,
        }

    # 6. Save compressed PyTorch tensor dataset
    out_file = PROCESSED_DIR / "full_dataset_cache.pt"
    print(f"\n[cache] Saving compressed dataset tensor cache to {out_file}...")
    torch.save(cache_dict, out_file)

    elapsed = round(time.time() - start_time, 2)
    print(f"\n[cache] [OK] Preprocessing complete in {elapsed}s!")
    print(f"[cache] Total cached patients: {len(cache_dict):,} (Train: {len(train_set):,}, Val: {len(val_set):,}, Test: {len(test_set):,})")


if __name__ == "__main__":
    build_and_cache_dataset()
