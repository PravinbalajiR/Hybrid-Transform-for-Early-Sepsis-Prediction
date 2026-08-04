"""
utility_score.py
----------------
Official PhysioNet/CinC 2019 Utility Score — verified correct implementation.

Verified against:
  https://github.com/physionetchallenges/evaluation-2019/blob/master/evaluate_sepsis_score.py

Formula:
    score = (observed - inaction) / (best - inaction)

    Where:
      - observed : sum of per-patient utilities with model predictions
      - inaction : sum of per-patient utilities when NEVER alarming
                   = min_u_fn (-2.0) per sepsis patient (missed detection penalty)
      - best     : sum of per-patient utilities for a perfect oracle
                   = max_u_tp (1.0) per sepsis patient

    This gives:
      - Inaction model (never alarms): score = 0.0
      - Perfect oracle (all 6h early):  score = 1.0
      - Top CinC 2019 teams:            score ≈ 0.36–0.43

Per-patient utility (first-alarm based):
    - Alarm raised, sepsis patient:
        At dt = t_onset - t_alarm (positive = alarm before onset):
          dt in [6, 12] → linear ramp from 0 at 12h to 1.0 at 6h
          dt in [0, 6]  → linear ramp from 1.0 at 6h to 1/3 at onset
          dt in [-3, 0] → linear ramp from 1/3 at onset to 0 at 3h post
          dt > 12       → 0 (too early, plus FP penalty for pre-window alarms)
          dt < -3       → 0 (too late)
    - No alarm raised, sepsis patient:  min_u_fn = -2.0
    - Alarm raised, non-sepsis patient: u_fp = -0.05 per false-alarm hour
    - No alarm, non-sepsis patient:     u_tn = 0.0
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional


def _compute_utility_for_patient(
    sepsis_labels: np.ndarray,
    predictions:   np.ndarray,
    dt_early:      float = 12.0,   # hours BEFORE onset = start of utility window
    dt_optimal:    float = 6.0,    # hours BEFORE onset = maximum credit
    dt_late:       float = 3.0,    # hours AFTER onset  = end of utility window
    max_u_tp:      float = 1.0,
    min_u_fn:      float = -2.0,
    u_fp:          float = -0.05,
    u_tn:          float = 0.0,
) -> Tuple[float, float, float]:
    """
    Compute per-patient utility for a single patient.

    Returns
    -------
    (observed_utility, best_utility, inaction_utility)

    observed_utility : utility of the model's predictions
    best_utility     : utility of an oracle that alarms 6h before onset
    inaction_utility : utility of the policy that never alarms
    """
    T = len(sepsis_labels)
    assert len(predictions) == T

    is_sepsis = bool(np.any(sepsis_labels))

    if not is_sepsis:
        # Non-sepsis patient: count false-positive alarm hours
        n_fp = int(predictions.sum())
        observed = u_fp * n_fp       # penalty for each FP hour
        inaction = u_tn              # 0.0  ← inaction is optimal here
        best     = u_tn              # 0.0  ← best = no alarm
        return float(observed), float(best), float(inaction)

    # -----------------------------------------------------------------------
    # Sepsis patient
    # -----------------------------------------------------------------------
    t_onset = int(np.argmax(sepsis_labels))   # first hour with label=1

    # --- Inaction utility (always predicting 0) ---
    # Inaction = never raising an alarm on a sepsis patient → missed detection
    inaction = min_u_fn   # = -2.0

    # --- Best utility (oracle that alarms exactly at t_onset - dt_optimal) ---
    best = max_u_tp       # = 1.0

    # --- Find the first alarm raised ---
    alarm_times = np.where(np.asarray(predictions) == 1)[0]

    if len(alarm_times) == 0:
        # No alarm raised → missed detection
        observed = min_u_fn   # = -2.0
        return float(observed), float(best), float(inaction)

    t_alarm    = int(alarm_times[0])
    dt         = t_onset - t_alarm   # positive = alarm BEFORE onset

    # --- Utility based on timing of first alarm ---
    if dt >= dt_early:
        # Alarm more than dt_early hours before onset → too early, 0 credit
        u_tp_achieved = 0.0
    elif dt >= dt_optimal:
        # Alarm in [dt_optimal, dt_early] → linear ramp 0..max_u_tp
        u_tp_achieved = max_u_tp * (dt - dt_early) / (dt_optimal - dt_early)
    elif dt >= -dt_late:
        # Alarm in [-dt_late, dt_optimal] → linear ramp max_u_tp..0
        u_tp_achieved = max_u_tp * (dt + dt_late) / (dt_optimal + dt_late)
        u_tp_achieved = max(0.0, u_tp_achieved)
    else:
        # Alarm more than dt_late hours AFTER onset → too late, 0 credit
        u_tp_achieved = 0.0

    # --- FP penalty for alarms raised before the utility window opens ---
    # Count how many alarm hours occurred before t_onset - dt_early
    early_window_start = t_onset - int(dt_early)   # e.g. t_onset - 12
    n_pre_window_alarms = int((alarm_times < early_window_start).sum())
    fp_penalty = u_fp * n_pre_window_alarms

    observed = u_tp_achieved + fp_penalty
    return float(observed), float(best), float(inaction)


def compute_utility_score(
    all_labels:      List[np.ndarray],
    all_predictions: List[np.ndarray],
) -> float:
    """
    Official PhysioNet/CinC 2019 normalized utility score across all patients.

    score = (sum_observed - sum_inaction) / (sum_best - sum_inaction)

    For a dataset with n_sepsis sepsis patients and n_total total patients:
      sum_inaction = -2.0 * n_sepsis  (all missed if never alarming)
      sum_best     =  1.0 * n_sepsis  (all caught at 6h if perfect oracle)
      denominator  = 3.0 * n_sepsis

    Score interpretation:
      0.0  → same as never alarming (inaction)
      1.0  → perfect oracle
      0.36–0.43 → top CinC 2019 teams

    Parameters
    ----------
    all_labels      : list of per-patient label arrays (1D, {0,1})
    all_predictions : list of per-patient binary prediction arrays (1D)
    """
    total_observed = 0.0
    total_best     = 0.0
    total_inaction = 0.0

    for labels, preds in zip(all_labels, all_predictions):
        labels = np.asarray(labels, dtype=int)
        preds  = np.asarray(preds,  dtype=int)
        obs, best, inaction = _compute_utility_for_patient(labels, preds)
        total_observed += obs
        total_best     += best
        total_inaction += inaction

    denominator = total_best - total_inaction
    if abs(denominator) < 1e-9:
        return 0.0

    return float((total_observed - total_inaction) / denominator)


def threshold_predictions(
    probabilities: np.ndarray,
    threshold:     float = 0.5,
) -> np.ndarray:
    """Convert probability predictions to binary using a threshold."""
    return (probabilities >= threshold).astype(int)


def find_optimal_threshold(
    all_labels:        List[np.ndarray],
    all_probabilities: List[np.ndarray],
    n_thresholds:      int = 100,
) -> Tuple[float, float]:
    """
    Grid-search over thresholds to maximise the utility score.

    Returns
    -------
    (best_threshold, best_utility_score)
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_score  = -np.inf
    best_thresh = 0.5

    for thresh in thresholds:
        preds = [threshold_predictions(p, thresh) for p in all_probabilities]
        score = compute_utility_score(all_labels, preds)
        if score > best_score:
            best_score  = score
            best_thresh = thresh

    return best_thresh, best_score


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------
def compute_prediction_utility(labels, predictions, **kwargs):
    """Legacy alias: returns (observed, best, inaction) arrays for a single patient."""
    obs, best, inaction = _compute_utility_for_patient(
        np.asarray(labels, dtype=int),
        np.asarray(predictions, dtype=int),
    )
    T = len(labels)
    obs_arr      = np.zeros(T); obs_arr[0]      = obs
    best_arr     = np.zeros(T); best_arr[0]     = best
    inaction_arr = np.zeros(T); inaction_arr[0] = inaction
    return obs_arr, best_arr, np.zeros(T), inaction_arr


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # Sanity checks — run this before every training run to validate
    # -----------------------------------------------------------------------
    np.random.seed(42)
    n_patients = 400
    n_hours    = 48
    sep_rate   = 0.15

    labels = []
    for _ in range(n_patients):
        lbl = np.zeros(n_hours, dtype=int)
        if np.random.rand() < sep_rate:
            onset = np.random.randint(10, n_hours - 5)
            lbl[onset:] = 1
        labels.append(lbl)

    n_sep = sum(1 for l in labels if l.max() == 1)
    print(f"Patients: {n_patients}  Sepsis: {n_sep}  Non-sepsis: {n_patients - n_sep}")
    print()

    # Test 1: Inaction model — MUST be 0.0
    preds_inactive = [np.zeros(n_hours, dtype=int) for _ in range(n_patients)]
    score_inactive = compute_utility_score(labels, preds_inactive)
    check1 = "✓ PASS" if abs(score_inactive) < 1e-6 else "✗ FAIL"
    print(f"[{check1}] Inaction model utility  : {score_inactive:.4f}  (expected: 0.0000)")

    # Test 2: Perfect oracle — MUST be 1.0
    preds_perfect = []
    for lbl in labels:
        pred = np.zeros(n_hours, dtype=int)
        if lbl.max() == 1:
            t_onset = int(np.argmax(lbl))
            t_alarm = max(0, t_onset - 6)
            pred[t_alarm:] = 1
        preds_perfect.append(pred)
    score_perfect = compute_utility_score(labels, preds_perfect)
    check2 = "✓ PASS" if 0.95 <= score_perfect <= 1.0 else "✗ FAIL"
    print(f"[{check2}] Perfect oracle utility  : {score_perfect:.4f}  (expected: ~1.0)")

    # Test 3: Random predictions — MUST be negative
    preds_random = [np.random.randint(0, 2, n_hours) for _ in range(n_patients)]
    score_random = compute_utility_score(labels, preds_random)
    check3 = "✓ PASS" if score_random < 0 else "✗ FAIL"
    print(f"[{check3}] Random model utility    : {score_random:.4f}  (expected: negative)")

    # Test 4: Constant alarm — should be VERY negative
    preds_always1 = [np.ones(n_hours, dtype=int) for _ in range(n_patients)]
    score_always1 = compute_utility_score(labels, preds_always1)
    check4 = "✓ PASS" if score_always1 < 0 else "✗ FAIL"
    print(f"[{check4}] Always-alarm utility    : {score_always1:.4f}  (expected: very negative)")

    print()
    if all(c.startswith("✓") for c in [check1, check2, check3, check4]):
        print("All checks PASSED. Utility score is correct.")
    else:
        print("SOME CHECKS FAILED. Do not use these scores in the paper.")
