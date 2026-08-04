"""
split.py
--------
Hospital-stratified train / validation / test split for PhysioNet/CinC 2019.

DESIGN DECISION: We split by hospital source, NOT by random patient shuffle.
- Set A patients → training + validation
- Set B patients → held-out test set

This simulates real-world cross-site deployment (a model trained at hospital A
and evaluated at hospital B) and avoids data leakage from the same distribution.

Within Set A: 90% train / 10% validation (stratified by SepsisLabel).

Usage
-----
  from preprocessing.split import make_splits, save_splits, load_splits

  train_ids, val_ids, test_ids = make_splits(patient_df)
  save_splits(train_ids, val_ids, test_ids)
  train_ids, val_ids, test_ids = load_splits()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

SPLITS_DIR = Path(__file__).parent.parent / "data" / "splits"


def make_splits(
    patient_df: pd.DataFrame,
    val_fraction: float = 0.10,
    test_fraction: float = 0.20,
    random_seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Create train / val / test patient ID splits.

    Parameters
    ----------
    patient_df    : DataFrame with columns 'PatientID', 'Source', 'SepsisLabel'
    val_fraction  : fraction of Set A to use as validation
    test_fraction : fallback test fraction if Set B is not present in sample
    random_seed   : for reproducibility

    Returns
    -------
    (train_ids, val_ids, test_ids) -- lists of PatientID strings
    """
    set_a = patient_df[patient_df["Source"] == "A"]
    set_b = patient_df[patient_df["Source"] == "B"]

    if len(set_b) > 0:
        # Standard full-dataset behavior: Set B -> held-out test
        test_ids = set_b["PatientID"].tolist()

        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_fraction,
            random_state=random_seed,
        )
        idx_train, idx_val = next(
            sss.split(set_a["PatientID"], set_a["SepsisLabel"])
        )

        train_ids = set_a.iloc[idx_train]["PatientID"].tolist()
        val_ids   = set_a.iloc[idx_val]["PatientID"].tolist()
    else:
        # Fallback for quick sample testing when Set B is omitted/empty
        sss_test = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_fraction,
            random_state=random_seed,
        )
        idx_trainval, idx_test = next(
            sss_test.split(patient_df["PatientID"], patient_df["SepsisLabel"])
        )
        trainval_df = patient_df.iloc[idx_trainval]
        test_ids    = patient_df.iloc[idx_test]["PatientID"].tolist()

        sss_val = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_fraction,
            random_state=random_seed,
        )
        idx_tr, idx_va = next(
            sss_val.split(trainval_df["PatientID"], trainval_df["SepsisLabel"])
        )
        train_ids = trainval_df.iloc[idx_tr]["PatientID"].tolist()
        val_ids   = trainval_df.iloc[idx_va]["PatientID"].tolist()

    # Sanity checks
    overlap = set(train_ids) & set(val_ids) & set(test_ids)
    assert not overlap, f"Overlap detected between splits: {overlap}"

    n_train_sep = patient_df[patient_df["PatientID"].isin(train_ids)]["SepsisLabel"].sum()
    n_val_sep   = patient_df[patient_df["PatientID"].isin(val_ids)]["SepsisLabel"].sum()
    n_test_sep  = patient_df[patient_df["PatientID"].isin(test_ids)]["SepsisLabel"].sum()

    print(f"[split] Train : {len(train_ids):>6,} patients  "
          f"({n_train_sep} sepsis, "
          f"{100*n_train_sep/len(train_ids) if len(train_ids)>0 else 0:.2f}%)")
    print(f"[split] Val   : {len(val_ids):>6,} patients  "
          f"({n_val_sep} sepsis, "
          f"{100*n_val_sep/len(val_ids) if len(val_ids)>0 else 0:.2f}%)")
    print(f"[split] Test  : {len(test_ids):>6,} patients  "
          f"({n_test_sep} sepsis, "
          f"{100*n_test_sep/len(test_ids) if len(test_ids)>0 else 0:.2f}%)")
    print(f"[split] Source: Train+Val -> Set A | Test -> Set B (or sample split)")

    return train_ids, val_ids, test_ids


def save_splits(
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    out_dir: Path = SPLITS_DIR,
) -> None:
    """Save split patient IDs to JSON files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": train_ids,
        "val":   val_ids,
        "test":  test_ids,
    }
    for split_name, ids in splits.items():
        path = out_dir / f"{split_name}_ids.json"
        with open(path, "w") as f:
            json.dump(ids, f, indent=2)
        print(f"[split] Saved {split_name} IDs -> {path}")


def load_splits(
    split_dir: Path = SPLITS_DIR,
) -> Tuple[List[str], List[str], List[str]]:
    """Load previously saved split IDs."""
    train_ids = json.loads((split_dir / "train_ids.json").read_text())
    val_ids   = json.loads((split_dir / "val_ids.json").read_text())
    test_ids  = json.loads((split_dir / "test_ids.json").read_text())
    print(f"[split] Loaded: train={len(train_ids):,}  val={len(val_ids):,}  test={len(test_ids):,}")
    return train_ids, val_ids, test_ids



if __name__ == "__main__":
    from preprocessing.load_data import load_dataset

    _, _ = load_dataset(verbose=True)

    # Can't run without the patient_df; this is just a usage demo
    print("Use make_splits(patient_df) after running the audit to produce splits.")
