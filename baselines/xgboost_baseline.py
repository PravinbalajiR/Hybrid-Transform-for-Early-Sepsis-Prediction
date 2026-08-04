"""
xgboost_baseline.py
--------------------
Model M1: XGBoost Baseline for Hourly Sepsis Prediction.

Strategy
--------
Loads pre-split and normalized tensors directly from full_dataset_cache.pt
(ensuring 100% identical splits and preprocessing as M2/M3/M4).

For each patient hour t, extracts cumulative features up to hour t:
  - Current observation values v_t, mask m_t, time-delta delta_t per variable
  - Cumulative mean, std, min, max up to hour t

Trains XGBoost to predict the hourly SepsisLabel and evaluates using the
official PhysioNet 2019 Utility Score and metrics.
"""

from __future__ import annotations

import sys
import os
import time
import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import xgboost as xgb
except ImportError:
    raise ImportError("Install xgboost: pip install xgboost")

from evaluation.utility_score import compute_utility_score, find_optimal_threshold
from evaluation.metrics import full_evaluation_report
from utils.seed import set_seed


def extract_hourly_features_from_sample(item: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract cumulative hourly feature matrix X (T, F_agg) and label vector y (T,) for one patient.
    Causally computes features up to hour t using NumPy cumulative operations.
    """
    v = item["values"].numpy()           # (T, F)
    m = item["mask"].numpy()             # (T, F)
    d = item["time_delta"].numpy()       # (T, F)
    y = item["labels"].numpy().astype(int) # (T,)
    
    T, F = v.shape
    
    v_obs = np.where(m == 1, v, 0.0)
    c_count = m.cumsum(axis=0)
    c_sum = v_obs.cumsum(axis=0)
    c_mean = np.where(c_count > 0, c_sum / np.maximum(c_count, 1.0), 0.0)
    
    c_sq = (v_obs ** 2).cumsum(axis=0)
    var = np.where(c_count > 1, (c_sq / np.maximum(c_count, 1.0)) - (c_mean ** 2), 0.0)
    c_std = np.sqrt(np.maximum(0.0, var))
    
    v_inf_min = np.where(m == 1, v, np.inf)
    c_min = np.minimum.accumulate(v_inf_min, axis=0)
    c_min = np.where(np.isinf(c_min), 0.0, c_min)
    
    v_inf_max = np.where(m == 1, v, -np.inf)
    c_max = np.maximum.accumulate(v_inf_max, axis=0)
    c_max = np.where(np.isinf(c_max), 0.0, c_max)
    
    X = np.hstack([v, m, d, c_mean, c_std, c_min, c_max]).astype(np.float32)
    return X, y


def build_dataset_from_cache(samples: list) -> Tuple[np.ndarray, np.ndarray, list, list]:
    X_list, y_list = [], []
    patient_labels, patient_X = [], []

    for item in tqdm(samples, desc="Building XGBoost features", leave=False):
        X_p, y_p = extract_hourly_features_from_sample(item)
        patient_labels.append(y_p)
        patient_X.append(X_p)
        X_list.append(X_p)
        y_list.append(y_p)

    X_flat = np.vstack(X_list)
    y_flat = np.concatenate(y_list)
    return X_flat, y_flat, patient_labels, patient_X


def train_and_evaluate() -> dict:
    set_seed(42)
    start_time = time.time()

    print("\n" + "=" * 60)
    print("  MODEL M1: XGBOOST BASELINE (HOURLY SEPSIS PREDICTION)")
    print("=" * 60)

    # 1. Load Precomputed Cached Dataset
    cache_path = Path(__file__).parent.parent / "data" / "processed" / "full_dataset_cache.pt"
    if not cache_path.exists():
        outer_cache_path = Path(__file__).parent.parent.parent / "processed" / "full_dataset_cache.pt"
        if outer_cache_path.exists():
            cache_path = outer_cache_path
        else:
            raise FileNotFoundError(f"Cached dataset not found at {cache_path} or {outer_cache_path}.")

    print(f"[xgboost] Loading cached dataset from {cache_path}...")
    cache_dict = torch.load(cache_path)

    train_samples, val_samples, test_samples = [], [], []

    for pid, item in cache_dict.items():
        item["patient_id"] = pid
        if item["split"] == "train":
            train_samples.append(item)
        elif item["split"] == "val":
            val_samples.append(item)
        else:
            test_samples.append(item)

    print(f"Split sizes: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")

    # 2. Extract hourly features
    print("\n[xgboost] Extracting hourly training features...")
    X_train, y_train, train_labels, train_X = build_dataset_from_cache(train_samples)
    print("\n[xgboost] Extracting hourly validation features...")
    X_val, y_val, val_labels, val_X = build_dataset_from_cache(val_samples)
    print("\n[xgboost] Extracting hourly test features...")
    X_test, y_test, test_labels, test_X = build_dataset_from_cache(test_samples)

    print(f"  Hourly Train shape: {X_train.shape}  pos={y_train.sum():,}")
    print(f"  Hourly Val shape  : {X_val.shape}    pos={y_val.sum():,}")
    print(f"  Hourly Test shape : {X_test.shape}   pos={y_test.sum():,}")

    # 3. Fit XGBoost
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

    # 4. Evaluation & Threshold Grid Search on Validation
    print("\n[xgboost] Evaluating on Validation set...")
    val_probas_list = [model.predict_proba(X_p)[:, 1] for X_p in val_X]
    best_thresh, best_val_utility = find_optimal_threshold(val_labels, val_probas_list, n_thresholds=20)
    print(f"  Best Validation Threshold: {best_thresh:.2f} | Utility Score: {best_val_utility:.4f}")

    # 5. Evaluate on Held-out Test Set
    print("\n[xgboost] Evaluating on Held-out Test Set...")
    inf_start = time.time()
    test_probas_list = [model.predict_proba(X_p)[:, 1] for X_p in test_X]
    inference_time = round(time.time() - inf_start, 2)

    test_preds_list = [(p >= best_thresh).astype(int) for p in test_probas_list]
    test_utility = compute_utility_score(test_labels, test_preds_list)

    report = full_evaluation_report(
        test_labels, test_probas_list, test_preds_list,
        utility_score=test_utility,
        split_name="test_xgboost",
    )

    exp_dir = Path(__file__).parent.parent / "experiments" / "xgboost" / "run_latest"
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=4)

    return report


if __name__ == "__main__":
    train_and_evaluate()
