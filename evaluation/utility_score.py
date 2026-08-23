"""
utility_score.py
----------------
Official PhysioNet/CinC 2019 Utility Score implementation wrapper.
Imports and executes the exact official PhysioNet 2019 challenge evaluator
from evaluation.official_physionet2019.py to guarantee 100% mathematical identity.

Reference:
  Reyna et al. (2019) "Early Prediction of Sepsis from Clinical Data:
  The PhysioNet/Computing in Cardiology Challenge 2019"
  https://physionet.org/content/challenge-2019/1.0.0/
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional
from evaluation.official_physionet2019 import compute_prediction_utility

def _compute_utility_for_patient(
    sepsis_labels: np.ndarray,
    predictions:   np.ndarray,
    dt_early:  float = -12.0,  # hours relative to t_sepsis
    dt_optimal: float = -6.0,  # hours relative to t_sepsis (t_sepsis = onset + 6)
    dt_late:   float = 3.0,    # hours relative to t_sepsis
    max_u_tp:  float = 1.0,
    min_u_fn:  float = -2.0,
    u_fp:      float = -0.05,
    u_tn:      float = 0.0,
) -> Tuple[float, float, float]:
    """
    Compute official PhysioNet 2019 per-patient observed utility, best utility, and inaction utility.
    """
    labels = np.asarray(sepsis_labels, dtype=int)
    preds  = np.asarray(predictions,   dtype=int)
    num_rows = len(labels)

    best_preds = np.zeros(num_rows, dtype=int)
    inact_preds = np.zeros(num_rows, dtype=int)

    if np.any(labels):
        t_sepsis = np.argmax(labels) - dt_optimal
        best_preds[max(0, int(t_sepsis + dt_early)) : min(int(t_sepsis + dt_late + 1), num_rows)] = 1

    obs_u   = compute_prediction_utility(labels, preds, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn, check_errors=False)
    best_u  = compute_prediction_utility(labels, best_preds, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn, check_errors=False)
    inact_u = compute_prediction_utility(labels, inact_preds, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn, check_errors=False)

    return obs_u, best_u, inact_u


def compute_utility_score(
    all_labels:      List[np.ndarray],
    all_predictions: List[np.ndarray],
) -> float:
    """
    Compute the official PhysioNet/CinC 2019 normalized utility score across all patients:
    U_normalized = (U_observed - U_inaction) / (U_best - U_inaction)
    """
    sum_obs   = 0.0
    sum_best  = 0.0
    sum_inact = 0.0

    for labels, preds in zip(all_labels, all_predictions):
        obs, best, inact = _compute_utility_for_patient(labels, preds)
        sum_obs   += obs
        sum_best  += best
        sum_inact += inact

    denom = sum_best - sum_inact
    if denom == 0:
        return 0.0

    return (sum_obs - sum_inact) / denom


def threshold_predictions(
    probabilities:  np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Convert probability predictions to binary using a threshold."""
    return (probabilities >= threshold).astype(int)


def find_optimal_threshold(
    all_labels:       List[np.ndarray],
    all_probabilities: List[np.ndarray],
    n_thresholds: int = 100,
) -> Tuple[float, float]:
    """
    Grid-search for the threshold that maximises the official PhysioNet utility score.
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_score = -np.inf
    best_thresh = 0.5

    for thresh in thresholds:
        preds = [threshold_predictions(p, thresh) for p in all_probabilities]
        score = compute_utility_score(all_labels, preds)
        if score > best_score:
            best_score = score
            best_thresh = thresh

    return best_thresh, best_score
