"""
load_data.py
------------
Utilities to load the PhysioNet/CinC 2019 dataset from the raw .psv files.

Each patient file contains hourly ICU observations as pipe-separated values.
This module provides:
  - load_patient_file()   : read a single .psv -> pandas DataFrame
  - load_dataset()        : load all patients from SetA and/or SetB
  - get_all_patient_files(): enumerate all .psv paths in a directory
"""

from __future__ import annotations

import os
import glob
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Column schema (41 columns, verified against actual dataset files)
# ---------------------------------------------------------------------------
VITAL_COLS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]

LAB_COLS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2",
    "AST", "BUN", "Alkalinephos", "Calcium", "Chloride",
    "Creatinine", "Bilirubin_direct", "Glucose", "Lactate",
    "Magnesium", "Phosphate", "Potassium", "Bilirubin_total",
    "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets",
]

DEMOGRAPHIC_COLS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]

LABEL_COL = "SepsisLabel"

ALL_FEATURE_COLS = VITAL_COLS + LAB_COLS
ALL_COLS = ALL_FEATURE_COLS + DEMOGRAPHIC_COLS + [LABEL_COL]

# ---------------------------------------------------------------------------
# Organ groupings — the "knowledge" layer
# ---------------------------------------------------------------------------
ORGAN_GROUPS = {
    "cardiovascular": ["HR", "SBP", "MAP", "DBP", "TroponinI"],
    "respiratory":    ["O2Sat", "Resp", "EtCO2", "FiO2", "PaCO2", "SaO2", "pH"],
    "renal":          ["Creatinine", "BUN", "Chloride", "Calcium",
                       "Potassium", "Magnesium", "Phosphate"],
    "liver":          ["AST", "Alkalinephos", "Bilirubin_direct", "Bilirubin_total"],
    "metabolic_hem":  ["Glucose", "Lactate", "BaseExcess", "HCO3",
                       "WBC", "Hct", "Hgb", "PTT", "Fibrinogen", "Platelets"],
    "temperature":    ["Temp"],
}

# Computed feature (derived, not in raw data)
DERIVED_FEATURES = {
    "ShockIndex": ("HR", "SBP"),   # HR / SBP
}


# ---------------------------------------------------------------------------
# Raw data paths (relative to the root of the cloned repo)
# ---------------------------------------------------------------------------
# Adjust these if your data lives elsewhere.
DEFAULT_SET_A = Path(__file__).parent.parent.parent / "training_setA" / "training"
DEFAULT_SET_B = Path(__file__).parent.parent.parent / "training_setB" / "training_setB"


def get_all_patient_files(directory: str | Path) -> List[Path]:
    """Return sorted list of .psv paths in *directory*."""
    directory = Path(directory)
    files = sorted(directory.glob("*.psv"))
    if not files:
        raise FileNotFoundError(
            f"No .psv files found in {directory}. "
            "Check that the path points to the correct training set folder."
        )
    return files


def load_patient_file(path: str | Path, patient_id: Optional[str] = None) -> pd.DataFrame:
    """
    Load a single per-patient .psv file into a DataFrame.

    Parameters
    ----------
    path        : path to the .psv file
    patient_id  : if None, inferred from the filename stem (e.g. 'p000001')

    Returns
    -------
    DataFrame with columns = ALL_COLS, plus 'PatientID'.
    NaN values represent missing observations (not imputed).
    """
    path = Path(path)
    pid = patient_id or path.stem

    df = pd.read_csv(path, sep="|", na_values=["NaN", "nan", ""], engine="python", on_bad_lines="skip")
    df.insert(0, "PatientID", pid)

    # Validate that all expected columns are present
    missing_cols = [c for c in ALL_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"File {path.name} is missing columns: {missing_cols}")

    return df


def load_dataset(
    set_a_dir: str | Path = DEFAULT_SET_A,
    set_b_dir: str | Path = DEFAULT_SET_B,
    include_set_a: bool = True,
    include_set_b: bool = True,
    max_patients: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[List[pd.DataFrame], pd.Series]:
    """
    Load the full PhysioNet 2019 dataset.

    Parameters
    ----------
    set_a_dir       : path to training_setA/training/
    set_b_dir       : path to training_setB/training_setB/
    include_set_a   : whether to load Set A
    include_set_b   : whether to load Set B
    max_patients    : if set, only load the first N patients (for quick testing)
    verbose         : show tqdm progress bar

    Returns
    -------
    patient_dfs : list of per-patient DataFrames (one per patient)
    source_map  : pd.Series  {PatientID -> 'A' | 'B'}
    """
    all_files: List[Tuple[Path, str]] = []

    if include_set_a:
        try:
            files_a = get_all_patient_files(set_a_dir)
            all_files.extend((f, "A") for f in files_a)
            if verbose:
                print(f"[load_dataset] Set A: {len(files_a):,} patients found")
        except FileNotFoundError as e:
            print(f"[WARNING] {e}")

    if include_set_b:
        try:
            files_b = get_all_patient_files(set_b_dir)
            all_files.extend((f, "B") for f in files_b)
            if verbose:
                print(f"[load_dataset] Set B: {len(files_b):,} patients found")
        except FileNotFoundError as e:
            print(f"[WARNING] {e}")

    if max_patients is not None:
        all_files = all_files[:max_patients]

    from concurrent.futures import ThreadPoolExecutor

    def _load_single(item: Tuple[Path, str]) -> Tuple[pd.DataFrame, str, str]:
        fpath, source = item
        df = load_patient_file(fpath)
        df["Source"] = source
        return df, df["PatientID"].iloc[0], source

    if verbose:
        print(f"[load_dataset] Loading {len(all_files):,} patients sequentially...")

    results = []
    for item in tqdm(all_files, desc="Loading patients", disable=not verbose):
        results.append(_load_single(item))

    patient_dfs = [r[0] for r in results]
    source_records = {r[1]: r[2] for r in results}

    source_map = pd.Series(source_records, name="Source")
    if verbose:
        print(f"[load_dataset] Total patients loaded: {len(patient_dfs):,}")

    return patient_dfs, source_map


def concat_all_patients(patient_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate per-patient DataFrames into one large DataFrame.
    Useful for computing global statistics. Memory-heavy for the full dataset.
    """
    return pd.concat(patient_dfs, ignore_index=True)


if __name__ == "__main__":
    # Quick smoke test: load 5 patients and print shapes
    dfs, src = load_dataset(max_patients=5, verbose=True)
    for df in dfs:
        pid = df["PatientID"].iloc[0]
        label = int(df["SepsisLabel"].max())
        print(f"  {pid}  rows={len(df)}  sepsis={label}  source={df['Source'].iloc[0]}")
