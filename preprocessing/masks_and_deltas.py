"""
masks_and_deltas.py
-------------------
Compute the two auxiliary inputs that turn a plain time-series into a
missingness-aware one:

  1. Observation Mask  M[t, j] = 1 if x[t,j] was observed, else 0
  2. Time Delta        Δ[t, j] = hours since variable j was last observed
                                 (set to 0 at t=0 if unobserved from start)

These are the missingness encoding described in:
  - Che et al. (2018)  "Recurrent Neural Networks for Multivariate Time Series
    with Missing Values"  (GRU-D)
  - Horn et al. (2020)  "Set Functions for Time Series"

Usage
-----
  from preprocessing.masks_and_deltas import compute_masks_and_deltas
  values, masks, deltas = compute_masks_and_deltas(patient_df, feature_cols)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Tuple


def compute_masks_and_deltas(
    patient_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For a single patient's DataFrame (T rows × F feature columns), produce:

    Parameters
    ----------
    patient_df   : DataFrame with hourly rows, columns ⊇ feature_cols.
                   NaN = missing observation.
    feature_cols : ordered list of feature column names to process.

    Returns
    -------
    values : np.ndarray  shape (T, F)  — raw values (NaN still present)
    masks  : np.ndarray  shape (T, F)  — 1 = observed, 0 = missing
    deltas : np.ndarray  shape (T, F)  — hours since last observation
    """
    X = patient_df[feature_cols].values.astype(np.float32)   # (T, F)
    T, F = X.shape

    masks  = (~np.isnan(X)).astype(np.float32)                # (T, F)
    deltas = np.zeros((T, F), dtype=np.float32)

    # t=0: delta is 0 for all variables (no prior observation)
    for t in range(1, T):
        for j in range(F):
            if masks[t - 1, j] == 1:
                # Variable was observed at t-1 → gap = 1 hour
                deltas[t, j] = 1.0
            else:
                # Variable was missing at t-1 → accumulate
                deltas[t, j] = deltas[t - 1, j] + 1.0

    return X, masks, deltas


def compute_masks_and_deltas_fast(
    patient_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast, exact implementation of compute_masks_and_deltas matching GRU-D convention.
    Ensures 100% equivalence with the reference recurrence function.
    """
    X = patient_df[feature_cols].values.astype(np.float32)   # (T, F)
    T, F = X.shape

    masks  = (~np.isnan(X)).astype(np.float32)
    deltas = np.zeros((T, F), dtype=np.float32)

    for j in range(F):
        d = 0.0
        for t in range(1, T):
            if masks[t - 1, j] == 1.0:
                d = 1.0
            else:
                d += 1.0
            deltas[t, j] = d

    return X, masks, deltas



def encode_triplet(
    values: np.ndarray,
    masks: np.ndarray,
    deltas: np.ndarray,
    impute_value: float = 0.0,
) -> np.ndarray:
    """
    Stack (value, mask, delta) into a (T, 3F) tensor — the input format used by
    the time-aware Transformer branch.

    Missing values are filled with *impute_value* (typically 0.0, i.e. the
    mean after z-score normalisation). The mask and delta still carry the
    original missingness information.

    Returns
    -------
    encoded : np.ndarray  shape (T, 3*F)
    """
    filled = np.where(masks == 1, values, impute_value)
    return np.concatenate([filled, masks, deltas], axis=-1)  # (T, 3F)


if __name__ == "__main__":
    # Quick unit test on patient p000001
    from preprocessing.load_data import load_patient_file, DEFAULT_SET_A, ALL_FEATURE_COLS

    fpath = next(DEFAULT_SET_A.glob("*.psv"))
    df = load_patient_file(fpath)
    print(f"Patient: {df['PatientID'].iloc[0]}  rows: {len(df)}")

    vals, m, d = compute_masks_and_deltas_fast(df, ALL_FEATURE_COLS)
    print(f"Values  shape: {vals.shape}")
    print(f"Masks   shape: {m.shape}  (mean observed: {m.mean():.3f})")
    print(f"Deltas  shape: {d.shape}  (mean gap: {d.mean():.2f} hrs)")

    encoded = encode_triplet(vals, m, d)
    print(f"Encoded shape: {encoded.shape}  (T × 3F = {vals.shape[1]} × 3)")
