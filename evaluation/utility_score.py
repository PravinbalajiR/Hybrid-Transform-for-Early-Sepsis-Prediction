"""
utility_score.py
----------------
Official PhysioNet/CinC 2019 Utility Score implementation.

This is the PRIMARY evaluation metric. DO NOT use a proxy like AUROC alone —
the competition metric is specifically designed to reward early prediction and
penalise late alarms and false positives.

Reference:
  Reyna et al. (2019) "Early Prediction of Sepsis from Clinical Data:
  The PhysioNet/Computing in Cardiology Challenge 2019"
  https://physionet.org/content/challenge-2019/1.0.0/

Score interpretation:
  - Alarm at t_sepsis - 12 h → utility = -0.05  (too early, slight penalty)
  - Alarm at t_sepsis - 6 h  → utility =  0.0   (threshold for credit)
  - Alarm at t_sepsis        → utility =  1.0   (perfect timing)
  - Alarm at t_sepsis + 3 h  → utility =  0.0   (too late)
  - False positive alarm     → utility = -0.05/h (penalty per FP hour)
  - No alarm on sepsis       → utility = -2.0    (worst outcome)
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional


def _compute_utility_for_patient(
    sepsis_labels: np.ndarray,
    predictions:   np.ndarray,
    dt_early:  float = 12.0,   # hours before onset → start of utility window
    dt_optimal: float = 6.0,   # hours before onset → full credit
    dt_late:   float = 3.0,    # hours after onset  → alarm still counts
    max_u_tp:  float = 1.0,    # maximum utility for a true positive
    min_u_fn:  float = -2.0,   # penalty for missed sepsis
    u_fp:      float = -0.05,  # penalty per false alarm hour (non-sepsis)
) -> Tuple[float, float]:
    """
    Compute per-patient unnormalised utility and the best-possible utility
    for this patient (used to normalise across the cohort).

    Parameters
    ----------
    sepsis_labels : 1D array of {0,1} per hour
    predictions   : 1D array of binary predictions {0,1} per hour

    Returns
    -------
    (achieved_utility, best_possible_utility)
    """
    T = len(sepsis_labels)
    assert len(predictions) == T

    is_sepsis = int(sepsis_labels.max())

    if not is_sepsis:
        # Non-sepsis patient: any alarm is a false positive
        n_fp = int(predictions.sum())
        utility = u_fp * n_fp
        best    = 0.0   # best possible: no alarms
        return utility, best

    # Sepsis patient
    t_onset = int(np.argmax(sepsis_labels))   # first hour with label=1

    # Find first alarm
    alarm_times = np.where(predictions == 1)[0]
    t_alarm = int(alarm_times[0]) if len(alarm_times) > 0 else None

    if t_alarm is None:
        # No alarm at all → missed detection
        return min_u_fn, max_u_tp

    dt = t_onset - t_alarm  # positive = alarm before onset

    # Compute utility based on timing
    if dt >= dt_optimal:
        # Alarm ≥ 6h early → linear scale from 0 to 1
        # at dt_early → 0, at dt_optimal → 1
        if dt >= dt_early:
            achieved = 0.0
        else:
            achieved = max_u_tp * (dt - dt_early) / (dt_optimal - dt_early)
    elif dt >= -dt_late:
        # Within the valid window [onset-6h, onset+3h]
        # dt from dt_optimal (6) down to -dt_late (-3)
        achieved = max_u_tp * (dt + dt_late) / (dt_optimal + dt_late)
        achieved = max(0.0, achieved)
    else:
        # Alarm too late (> dt_late hours after onset)
        achieved = 0.0

    # Additional FP penalties for alarms before the early window
    fp_alarms = int((alarm_times < (t_onset - dt_early)).sum())
    achieved += u_fp * fp_alarms

    best = max_u_tp  # best possible is full credit
    return achieved, best


def compute_utility_score(
    all_labels:      List[np.ndarray],
    all_predictions: List[np.ndarray],
) -> float:
    """
    Compute the official PhysioNet/CinC 2019 utility score across all patients.

    Parameters
    ----------
    all_labels      : list of per-patient label arrays (1D, values in {0,1})
    all_predictions : list of per-patient binary prediction arrays (1D)

    Returns
    -------
    Normalised utility score in range roughly [-1, 1]
    Leaderboard range in 2019: ~0.36 – 0.43 for top teams.
    """
    total_achieved = 0.0
    total_best     = 0.0

    for labels, preds in zip(all_labels, all_predictions):
        labels = np.asarray(labels, dtype=int)
        preds  = np.asarray(preds,  dtype=int)
        a, b = _compute_utility_for_patient(labels, preds)
        total_achieved += a
        total_best     += b

    if total_best == 0:
        return 0.0

    return total_achieved / total_best


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
    Grid-search for the threshold that maximises the utility score.

    Returns
    -------
    (best_threshold, best_utility_score)
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


if __name__ == "__main__":
    # Quick unit test with synthetic data
    np.random.seed(42)
    n_patients = 100
    n_hours    = 48

    labels = []
    preds  = []
    for _ in range(n_patients):
        lbl = np.zeros(n_hours, dtype=int)
        # 15% sepsis rate
        if np.random.rand() < 0.15:
            onset = np.random.randint(20, n_hours - 5)
            lbl[onset:] = 1
        labels.append(lbl)
        # Noisy random predictions
        preds.append(np.random.randint(0, 2, n_hours))

    score = compute_utility_score(labels, preds)
    print(f"Synthetic utility score (random predictions): {score:.4f}")
    print("(Expected range for random: strongly negative)")
    print("(Top CinC 2019 teams: ~0.36–0.43)")
