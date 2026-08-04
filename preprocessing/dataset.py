"""
dataset.py
----------
PyTorch Dataset for PhysioNet/CinC 2019 Sepsis Prediction.

Serves as the core data engine for all PyTorch models (Plain Transformer,
Time-Aware Transformer, Organ Branch, and Hybrid Transformer).

Each item returns a dictionary containing:
  - 'values'     : FloatTensor (T, F)  -- Z-score normalized (NaNs filled with 0)
  - 'mask'       : FloatTensor (T, F)  -- 1 if observed, 0 if missing
  - 'time_delta' : FloatTensor (T, F)  -- hours since last observation
  - 'triplet'    : FloatTensor (T, 3*F)-- concatenated [values * mask, mask, time_delta]
  - 'labels'     : FloatTensor (T,)    -- hourly binary SepsisLabel
  - 'length'     : LongTensor  ()      -- actual sequence length (unpadded)
  - 'patient_id' : str                 -- Patient ID
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any

from preprocessing.load_data import ALL_FEATURE_COLS, LABEL_COL
from preprocessing.masks_and_deltas import compute_masks_and_deltas_fast, encode_triplet


class SepsisDataset(Dataset):
    """
    PyTorch Dataset returning irregular time-series triplets per patient.
    """

    def __init__(
        self,
        patient_dfs: List[pd.DataFrame],
        feature_cols: Optional[List[str]] = None,
        max_len: Optional[int] = 168,  # cap sequence length at e.g. 7 days (168 hours)
        impute_value: float = 0.0,     # mean after z-score normalization
    ):
        self.feature_cols = feature_cols or ALL_FEATURE_COLS
        self.max_len = max_len
        self.impute_value = impute_value

        self.samples: List[Dict[str, Any]] = []
        self._prepare_samples(patient_dfs)

    def _prepare_samples(self, patient_dfs: List[pd.DataFrame]) -> None:
        """Pre-compute features, masks, and time-deltas for fast retrieval."""
        for df in patient_dfs:
            pid = df["PatientID"].iloc[0]

            # Truncate sequence to max_len if specified
            if self.max_len and len(df) > self.max_len:
                df_curr = df.iloc[:self.max_len].copy()
            else:
                df_curr = df

            X, masks, deltas = compute_masks_and_deltas_fast(df_curr, self.feature_cols)
            labels = df_curr[LABEL_COL].values.astype(np.float32)
            seq_len = len(df_curr)

            # Impute values array for model input (0.0 represents mean post-normalization)
            values_imputed = np.where(masks == 1.0, X, self.impute_value)

            # Form concatenated triplet: (T, 3*F)
            triplet = encode_triplet(X, masks, deltas, impute_value=self.impute_value)

            self.samples.append({
                "patient_id": pid,
                "values":     torch.tensor(values_imputed, dtype=torch.float32),
                "mask":       torch.tensor(masks,          dtype=torch.float32),
                "time_delta": torch.tensor(deltas,         dtype=torch.float32),
                "triplet":    torch.tensor(triplet,        dtype=torch.float32),
                "labels":     torch.tensor(labels,         dtype=torch.float32),
                "length":     torch.tensor(seq_len,        dtype=torch.long),
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def collate_sepsis_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function to pad variable-length patient time-series into fixed-size tensors.

    Pads sequence dimension (dim 0) with zeros up to max length in the batch.
    Returns padded tensors and an explicit sequence mask (dim B x T).
    """
    batch_size = len(batch)
    max_t = max(item["length"].item() for item in batch)
    num_features = batch[0]["values"].shape[1]

    padded_values     = torch.zeros(batch_size, max_t, num_features, dtype=torch.float32)
    padded_mask       = torch.zeros(batch_size, max_t, num_features, dtype=torch.float32)
    padded_delta      = torch.zeros(batch_size, max_t, num_features, dtype=torch.float32)
    padded_triplet    = torch.zeros(batch_size, max_t, 3 * num_features, dtype=torch.float32)
    padded_labels     = torch.zeros(batch_size, max_t, dtype=torch.float32)
    seq_lengths       = torch.zeros(batch_size, dtype=torch.long)
    padding_mask      = torch.ones(batch_size, max_t, dtype=torch.bool)  # True = padded

    patient_ids = []

    for i, item in enumerate(batch):
        t = item["length"].item()
        padded_values[i, :t]  = item["values"]
        padded_mask[i, :t]    = item["mask"]
        padded_delta[i, :t]   = item["time_delta"]
        padded_triplet[i, :t] = item["triplet"]
        padded_labels[i, :t]  = item["labels"]
        seq_lengths[i]        = t
        padding_mask[i, :t]   = False  # False = valid observation
        patient_ids.append(item["patient_id"])

    return {
        "values":       padded_values,     # (B, T, F)
        "mask":         padded_mask,       # (B, T, F)
        "time_delta":   padded_delta,      # (B, T, F)
        "triplet":      padded_triplet,    # (B, T, 3F)
        "labels":       padded_labels,     # (B, T)
        "lengths":      seq_lengths,       # (B,)
        "padding_mask": padding_mask,      # (B, T) True for padded positions
        "patient_ids":  patient_ids,
    }


def create_dataloader(
    patient_dfs: List[pd.DataFrame],
    batch_size: int = 32,
    shuffle: bool = True,
    max_len: Optional[int] = 168,
    num_workers: int = 0,
) -> DataLoader:
    """Convenience helper to build PyTorch DataLoader."""
    dataset = SepsisDataset(patient_dfs, max_len=max_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_sepsis_batch,
        num_workers=num_workers,
    )


class CachedSepsisDataset(Dataset):
    """Dataset that wraps precomputed and cached tensors from full_dataset_cache.pt."""
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def create_cached_dataloader(
    samples: List[Dict[str, Any]],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = None,
) -> DataLoader:
    dataset = CachedSepsisDataset(samples)
    
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "collate_fn": collate_sepsis_batch,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
            
    return DataLoader(dataset, **kwargs)


if __name__ == "__main__":
    from preprocessing.load_data import load_dataset

    print("[Dataset Test] Loading 5 patients...")
    dfs, _ = load_dataset(max_patients=5, verbose=False)
    dataset = SepsisDataset(dfs)
    print(f"Dataset length: {len(dataset)}")

    loader = create_dataloader(dfs, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    print("\n[Batch Structure]")
    print(f"  values shape      : {batch['values'].shape}")
    print(f"  mask shape        : {batch['mask'].shape}")
    print(f"  time_delta shape  : {batch['time_delta'].shape}")
    print(f"  triplet shape     : {batch['triplet'].shape} (Values + Mask + Delta)")
    print(f"  labels shape      : {batch['labels'].shape}")
    print(f"  padding_mask shape: {batch['padding_mask'].shape}")
