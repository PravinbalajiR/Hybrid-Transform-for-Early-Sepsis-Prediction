"""
xgboost_baseline.py
--------------------
Model M1: XGBoost Baseline for Hourly Sepsis Prediction.

Strategy
--------
For each patient hour t, extract cumulative features up to hour t:
  - Last observed value, cumulative mean, std, min, max, missingness count
  - Current observation mask m_t and time-delta delta_t per variable
  - Demographics (Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS)

Trains XGBoost to predict the hourly SepsisLabel and evaluates using the
official PhysioNet 2019 Utility Score and threshold grid search.
"""

from __future__ import annotations

import sys
import os
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
except ImportError:
    raise ImportError("Install xgboost: pip install xgboost")

from preprocessing.load_data import (
    load_dataset, ALL_FEATURE_COLS, DEMOGRAPHIC_COLS, LABEL_COL, DEFAULT_SET_A, DEFAULT_SET_B
)
from preprocessing.split import make_splits, save_splits, load_splits
from preprocessing.masks_and_deltas import compute_masks_and_deltas_fast
from preprocessing.normalize import Normalizer
from evaluation.utility_score import compute_utility_score, find_optimal_threshold
from evaluation.metrics import compute_classification_metrics, full_evaluation_report
from utils.seed import set_seed
from utils.logger import ExperimentLogger


def extract_hourly_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract cumulative hourly feature matrix X (T, F_agg) and label vector y (T,) for one patient.
    Vectorized using NumPy cumulative operations for fast processing.
    """
    T = len(df)
    vals, masks, deltas = compute_masks_and_deltas_fast(df, ALL_FEATURE_COLS)
    labels = df[LABEL_COL].values.astype(int)
    F = vals.shape[1]

    v_zero = np.where(masks == 1, vals, 0.0)
    c_count = masks.cumsum(axis=0)
    c_sum = v_zero.cumsum(axis=0)
    c_mean = np.where(c_count > 0, c_sum / np.maximum(c_count, 1.0), 0.0)

    c_sq = (v_zero ** 2).cumsum(axis=0)
    var = np.where(c_count > 1, (c_sq / np.maximum(c_count, 1.0)) - (c_mean ** 2), 0.0)
    c_std = np.sqrt(np.maximum(0.0, var))

    val_inf_min = np.where(masks == 1, vals, np.inf)
    c_min = np.minimum.accumulate(val_inf_min, axis=0)
    c_min = np.where(np.isinf(c_min), 0.0, c_min)

    val_inf_max = np.where(masks == 1, vals, -np.inf)
    c_max = np.maximum.accumulate(val_inf_max, axis=0)
    c_max = np.where(np.isinf(c_max), 0.0, c_max)

    cur_vals = np.where(masks == 1, vals, 0.0)
    stats = np.stack([c_mean, c_std, c_min, c_max], axis=-1).reshape(T, 4 * F)

    dem_vals = np.tile([
        df["Age"].iloc[0],
        df["Gender"].iloc[0],
        df["Unit1"].iloc[0] if pd.notna(df["Unit1"].iloc[0]) else 0.0,
        df["Unit2"].iloc[0] if pd.notna(df["Unit2"].iloc[0]) else 0.0,
        df["HospAdmTime"].iloc[0],
    ], (T, 1))

    iculos = df["ICULOS"].values.astype(np.float32)[:, None]
    X = np.hstack([cur_vals, masks, deltas, stats, dem_vals, iculos]).astype(np.float32)
    return X, labels


def build_hourly_dataset(
    patient_dfs: List[pd.DataFrame],
    ids_to_use: set[str],
    subsample_rate: int = 1,  # use every Nth hour for training if memory-constrained
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    Build hourly feature dataset for training and evaluation.

    Returns:
      X_flat : 2D array of all stacked hourly features
      y_flat : 1D array of all stacked hourly labels
      patient_labels : list of 1D arrays per patient
      patient_X      : list of 2D feature matrices per patient
    """
    X_list, y_list = [], []
    patient_labels, patient_X = [], []

    for df in tqdm(patient_dfs, desc="Building hourly features", leave=False):
        pid = df["PatientID"].iloc[0]
        if pid not in ids_to_use:
            continue

        X_p, y_p = extract_hourly_features(df)
        patient_labels.append(y_p)
        patient_X.append(X_p)

        if subsample_rate > 1:
            X_list.append(X_p[::subsample_rate])
            y_list.append(y_p[::subsample_rate])
        else:
            X_list.append(X_p)
            y_list.append(y_p)

    X_flat = np.vstack(X_list)
    y_flat = np.concatenate(y_list)
    return X_flat, y_flat, patient_labels, patient_X


def train_and_evaluate(max_patients: Optional[int] = None) -> dict:
    set_seed(42)
    start_time = time.time()

    print("\n" + "=" * 60)
    print("  MODEL M1: XGBOOST BASELINE (HOURLY SEPSIS PREDICTION)")
    print("=" * 60)

    # 1. Load dataset
    patient_dfs, _ = load_dataset(
        set_a_dir=DEFAULT_SET_A,
        set_b_dir=DEFAULT_SET_B,
        max_patients=max_patients,
        verbose=True,
    )

    # 2. Split
    splits_dir = Path(__file__).parent.parent / "data" / "splits"
    patient_records = [
        {
            "PatientID":   df["PatientID"].iloc[0],
            "Source":      df["Source"].iloc[0],
            "SepsisLabel": int(df[LABEL_COL].max()),
        }
        for df in patient_dfs
    ]
    patient_df = pd.DataFrame(patient_records)

    if (splits_dir / "train_ids.json").exists():
        train_ids, val_ids, test_ids = load_splits(splits_dir)
    else:
        train_ids, val_ids, test_ids = make_splits(patient_df)
        save_splits(train_ids, val_ids, test_ids, splits_dir)

    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)

    dfs_train = [df for df in patient_dfs if df["PatientID"].iloc[0] in train_set]
    dfs_val   = [df for df in patient_dfs if df["PatientID"].iloc[0] in val_set]
    dfs_test  = [df for df in patient_dfs if df["PatientID"].iloc[0] in test_set]

    print(f"Split sizes: Train={len(dfs_train)}, Val={len(dfs_val)}, Test={len(dfs_test)}")

    # 3. Z-score Normalization (Fit on Train only!)
    norm = Normalizer()
    norm.fit(dfs_train, ALL_FEATURE_COLS, verbose=False)
    dfs_train = norm.transform(dfs_train)
    dfs_val   = norm.transform(dfs_val)
    dfs_test  = norm.transform(dfs_test)

    # 4. Extract hourly features
    print("\n[xgboost] Extracting hourly training features...")
    X_train, y_train, train_labels, train_X = build_hourly_dataset(dfs_train, train_set, subsample_rate=1)
    print("\n[xgboost] Extracting hourly validation features...")
    X_val, y_val, val_labels, val_X = build_hourly_dataset(dfs_val, val_set, subsample_rate=1)
    print("\n[xgboost] Extracting hourly test features...")
    X_test, y_test, test_labels, test_X = build_hourly_dataset(dfs_test, test_set, subsample_rate=1)

    print(f"  Hourly Train shape: {X_train.shape}  pos={y_train.sum():,}")
    print(f"  Hourly Val shape  : {X_val.shape}    pos={y_val.sum():,}")
    print(f"  Hourly Test shape : {X_test.shape}   pos={y_test.sum():,}")

    # 5. Fit XGBoost
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pw = neg / pos if pos > 0 else 1.0
    print(f"\n[xgboost] Training XGBoost (scale_pos_weight={scale_pw:.1f})...")

    train_start = time.time()
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pw,
        eval_metric="auc",
        early_stopping_rounds=25,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    train_time = round(time.time() - train_start, 2)

    # 6. Evaluation & Threshold Grid Search on Validation
    print("\n[xgboost] Evaluating on Validation set...")
    val_probas_list = [model.predict_proba(X_p)[:, 1] for X_p in val_X]
    best_thresh, best_val_utility = find_optimal_threshold(val_labels, val_probas_list, n_thresholds=20)
    print(f"  Best Validation Threshold: {best_thresh:.2f} | Utility Score: {best_val_utility:.4f}")

    # 7. Evaluate on Held-out Test Set B
    print("\n[xgboost] Evaluating on Held-out Test Set B...")
    inf_start = time.time()
    test_probas_list = [model.predict_proba(X_p)[:, 1] for X_p in test_X]
    inference_time = round(time.time() - inf_start, 2)

    test_preds_list = [(p >= best_thresh).astype(int) for p in test_probas_list]
    test_utility = compute_utility_score(test_labels, test_preds_list)

    report = full_evaluation_report(
        test_labels, test_probas_list, test_preds_list,
        utility_score=test_utility,
        split_name="test_set_b_xgboost",
    )

    logger = ExperimentLogger(
        log_dir=Path(__file__).parent.parent / "experiments" / "logs",
        experiment_name="xgboost_baseline",
    )
    logger.summary({
        **report,
        "train_time_sec": train_time,
        "inference_time_sec": inference_time,
    })

    return report


if __name__ == "__main__":
    train_and_evaluate()
