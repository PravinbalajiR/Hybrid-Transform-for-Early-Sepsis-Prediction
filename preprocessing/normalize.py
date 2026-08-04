"""
normalize.py
------------
Z-score normalization for the PhysioNet/CinC 2019 features.

CRITICAL RULE: Statistics (mean, std) are computed ONLY on the training split
and then applied to validation and test splits — never fit on val/test.

Handles NaN correctly: NaN values are ignored during stat computation and
left as NaN after normalization (the mask layer handles them downstream).

Usage
-----
  from preprocessing.normalize import Normalizer

  norm = Normalizer()
  norm.fit(train_dfs, feature_cols)
  train_dfs_norm = norm.transform(train_dfs, feature_cols)
  val_dfs_norm   = norm.transform(val_dfs,   feature_cols)

  norm.save("data/processed/normalizer.pkl")
  norm2 = Normalizer.load("data/processed/normalizer.pkl")
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm


class Normalizer:
    """
    Per-variable z-score normalization.
    Fit on training data only. Handles NaN-sparse ICU data safely.
    """

    def __init__(self):
        self.means_: Optional[Dict[str, float]] = None
        self.stds_:  Optional[Dict[str, float]] = None
        self.feature_cols_: Optional[List[str]] = None
        self._fitted = False

    def fit(
        self,
        patient_dfs: List[pd.DataFrame],
        feature_cols: List[str],
        verbose: bool = True,
    ) -> "Normalizer":
        """
        Compute mean and std from all observed values in the training set.

        Parameters
        ----------
        patient_dfs  : list of per-patient DataFrames (training split only)
        feature_cols : columns to normalize
        """
        self.feature_cols_ = feature_cols

        # Stack all values into one array to compute global stats
        # Use incremental approach to avoid OOM on 40k patients
        sums   = {col: 0.0 for col in feature_cols}
        sq_sums = {col: 0.0 for col in feature_cols}
        counts  = {col: 0   for col in feature_cols}

        for df in tqdm(patient_dfs, desc="Fitting normalizer", disable=not verbose):
            for col in feature_cols:
                vals = df[col].dropna().values
                n = len(vals)
                if n == 0:
                    continue
                sums[col]    += vals.sum()
                sq_sums[col] += (vals ** 2).sum()
                counts[col]  += n

        self.means_ = {}
        self.stds_  = {}
        for col in feature_cols:
            n = counts[col]
            if n == 0:
                self.means_[col] = 0.0
                self.stds_[col]  = 1.0
            else:
                mu = sums[col] / n
                var = sq_sums[col] / n - mu ** 2
                self.means_[col] = mu
                self.stds_[col]  = max(np.sqrt(var), 1e-6)  # avoid divide-by-zero

        self._fitted = True
        return self

    def transform(
        self,
        patient_dfs: List[pd.DataFrame],
        feature_cols: Optional[List[str]] = None,
        inplace: bool = False,
    ) -> List[pd.DataFrame]:
        """
        Apply z-score normalization to a list of patient DataFrames.
        NaN values remain NaN.

        Parameters
        ----------
        patient_dfs  : list of per-patient DataFrames
        feature_cols : columns to transform (defaults to those used in fit)
        inplace      : modify DataFrames in place (saves memory)

        Returns
        -------
        List of normalized DataFrames
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .transform()")

        cols = feature_cols or self.feature_cols_
        result = []

        for df in patient_dfs:
            out = df if inplace else df.copy()
            for col in cols:
                if col in out.columns:
                    mu  = self.means_[col]
                    std = self.stds_[col]
                    out[col] = (out[col] - mu) / std
            result.append(out)

        return result

    def fit_transform(
        self,
        patient_dfs: List[pd.DataFrame],
        feature_cols: List[str],
    ) -> List[pd.DataFrame]:
        """Convenience: fit then transform in one call."""
        self.fit(patient_dfs, feature_cols)
        return self.transform(patient_dfs, feature_cols)

    def inverse_transform_value(self, col: str, z: float) -> float:
        """Convert a z-score back to original scale for a single variable."""
        return z * self.stds_[col] + self.means_[col]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[Normalizer] Saved -> {path}")

    @classmethod
    def load(cls, path: str | Path) -> "Normalizer":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        assert isinstance(obj, cls)
        return obj

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame of mean and std per variable."""
        if not self._fitted:
            raise RuntimeError("Not yet fitted.")
        return pd.DataFrame({
            "Variable": list(self.means_.keys()),
            "Mean":     list(self.means_.values()),
            "Std":      list(self.stds_.values()),
        })


if __name__ == "__main__":
    from preprocessing.load_data import load_dataset, ALL_FEATURE_COLS

    dfs, _ = load_dataset(max_patients=200, verbose=True)

    norm = Normalizer()
    norm.fit(dfs, ALL_FEATURE_COLS, verbose=False)
    print(norm.summary().to_string(index=False))
