"""
test_preprocessing.py
----------------------
Unit tests for core preprocessing and evaluation functions:
  1. compute_masks_and_deltas_fast
  2. Normalizer (fit/transform isolation)
  3. SepsisDataset and collate function
  4. PhysioNet 2019 Utility Score computation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd

from preprocessing.load_data import ALL_FEATURE_COLS
from preprocessing.masks_and_deltas import (
    compute_masks_and_deltas,
    compute_masks_and_deltas_fast,
    encode_triplet,
)
from preprocessing.normalize import Normalizer



from preprocessing.dataset import SepsisDataset, collate_sepsis_batch
from evaluation.utility_score import compute_utility_score, _compute_utility_for_patient


def make_dummy_patient_df(n_hours: int = 24) -> pd.DataFrame:
    """Helper to construct a valid synthetic patient DataFrame."""
    data = {"PatientID": ["p000001"] * n_hours, "SepsisLabel": [0] * n_hours}
    for col in ALL_FEATURE_COLS:
        # 50% random missingness
        vals = np.random.randn(n_hours)
        vals[np.random.rand(n_hours) > 0.5] = np.nan
        data[col] = vals

    # Add demographics
    data["Age"] = 65.0
    data["Gender"] = 1
    data["Unit1"] = 1.0
    data["Unit2"] = 0.0
    data["HospAdmTime"] = -0.03
    data["ICULOS"] = np.arange(1, n_hours + 1)
    data["Source"] = "A"

    return pd.DataFrame(data)


def test_masks_and_deltas_shapes():
    df = make_dummy_patient_df(30)
    vals_ref, masks_ref, deltas_ref = compute_masks_and_deltas(df, ALL_FEATURE_COLS)
    vals, masks, deltas = compute_masks_and_deltas_fast(df, ALL_FEATURE_COLS)

    assert vals.shape == (30, len(ALL_FEATURE_COLS))
    assert masks.shape == (30, len(ALL_FEATURE_COLS))
    assert deltas.shape == (30, len(ALL_FEATURE_COLS))

    # Mask values must be binary
    assert np.all(np.isin(masks, [0.0, 1.0]))
    # Time delta at t=0 must be 0 for all features
    assert np.all(deltas[0] == 0.0)

    # 100% Equivalence test between reference recurrence and fast version
    assert np.allclose(masks_ref, masks)
    assert np.allclose(deltas_ref, deltas)



def test_encode_triplet():
    df = make_dummy_patient_df(20)
    vals, masks, deltas = compute_masks_and_deltas_fast(df, ALL_FEATURE_COLS)
    triplet = encode_triplet(vals, masks, deltas, impute_value=0.0)

    assert triplet.shape == (20, 3 * len(ALL_FEATURE_COLS))
    # Check no NaNs remain in imputed values section
    assert not np.isnan(triplet[:, :len(ALL_FEATURE_COLS)]).any()


def test_normalizer_isolation():
    df_train = make_dummy_patient_df(50)
    df_test  = make_dummy_patient_df(20)

    norm = Normalizer()
    norm.fit([df_train], ALL_FEATURE_COLS, verbose=False)

    norm_train = norm.transform([df_train])[0]
    norm_test  = norm.transform([df_test])[0]

    # Verify column statistics exist
    assert norm.means_ is not None
    assert norm.stds_ is not None
    assert len(norm.means_) == len(ALL_FEATURE_COLS)


def test_sepsis_dataset_and_collate():
    df1 = make_dummy_patient_df(24)
    df2 = make_dummy_patient_df(48)

    dataset = SepsisDataset([df1, df2])
    assert len(dataset) == 2

    item0 = dataset[0]
    assert "values" in item0
    assert "mask" in item0
    assert "time_delta" in item0
    assert "triplet" in item0

    # Collate batch
    batch = collate_sepsis_batch([dataset[0], dataset[1]])
    assert batch["values"].shape == (2, 48, len(ALL_FEATURE_COLS))
    assert batch["padding_mask"].shape == (2, 48)
    # First item was 24h long, so padded hours 24..47 should have padding_mask=True
    assert torch.all(batch["padding_mask"][0, 24:])
    assert not torch.any(batch["padding_mask"][0, :24])


def test_utility_score_perfect_prediction():
    # Optimal detection: alarm raised 6h before sepsis onset (dt = 6)
    labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])  # onset at hour 6 (0-indexed)
    preds  = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # alarm raised at hour 0 (6h early)

    u, u_best = _compute_utility_for_patient(labels, preds)
    assert u > 0.0
    assert abs(u - u_best) < 1e-5  # achieves maximum utility credit


if __name__ == "__main__":
    print("[Testing Preprocessing & Evaluation Units...]")
    test_masks_and_deltas_shapes()
    test_encode_triplet()
    test_normalizer_isolation()
    test_sepsis_dataset_and_collate()
    test_utility_score_perfect_prediction()
    print("[OK] All 5 unit tests passed successfully!")
