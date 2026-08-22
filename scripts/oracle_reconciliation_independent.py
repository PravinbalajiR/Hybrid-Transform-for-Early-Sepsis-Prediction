"""
oracle_reconciliation_independent.py
--------------------------------------
Zero-dependency, standalone utility and oracle calculator for PhysioNet 2019 metric.
Uses ONLY NumPy and Pandas. Zero imports from model code, utility scorer, or checkpoints.
"""

import numpy as np
import pandas as pd

def calculate_patient_utility(labels: np.ndarray, preds: np.ndarray, dt_early=12.0, dt_optimal=6.0, dt_late=3.0, max_u_tp=1.0, min_u_fn=-2.0, u_fp=-0.05):
    """Computes (achieved_utility, best_possible_utility) for a single patient."""
    labels = np.asarray(labels, dtype=int)
    preds = np.asarray(preds, dtype=int)
    is_sepsis = int(labels.max()) == 1

    if not is_sepsis:
        fp_hours = int(preds.sum())
        return u_fp * fp_hours, 0.0

    t_onset = int(np.argmax(labels))
    alarm_indices = np.where(preds == 1)[0]
    if len(alarm_indices) == 0:
        return min_u_fn, max_u_tp

    t_alarm = int(alarm_indices[0])
    dt = t_onset - t_alarm

    if dt >= dt_optimal:
        if dt >= dt_early: achieved = 0.0
        else: achieved = max_u_tp * (dt - dt_early) / (dt_optimal - dt_early)
    elif dt >= -dt_late:
        achieved = max_u_tp * (dt + dt_late) / (dt_optimal + dt_late)
        achieved = max(0.0, achieved)
    else:
        achieved = 0.0

    fp_alarms = int((alarm_indices < (t_onset - dt_early)).sum())
    achieved += u_fp * fp_alarms
    return achieved, max_u_tp


def calculate_never_alarm(labels: np.ndarray):
    """Computes prediction and utility when never issuing an alarm."""
    preds = np.zeros(len(labels), dtype=int)
    achieved, best = calculate_patient_utility(labels, preds)
    return preds, achieved, best


def calculate_always_alarm(labels: np.ndarray):
    """Computes prediction and utility when always issuing an alarm (1 at every hour)."""
    preds = np.ones(len(labels), dtype=int)
    achieved, best = calculate_patient_utility(labels, preds)
    return preds, achieved, best


def calculate_onset_alarm(labels: np.ndarray):
    """Computes prediction and utility when issuing an alarm exactly at sepsis onset."""
    preds = np.zeros(len(labels), dtype=int)
    if labels.max() == 1:
        t_onset = int(np.argmax(labels))
        preds[t_onset] = 1
    achieved, best = calculate_patient_utility(labels, preds)
    return preds, achieved, best


def calculate_best_single_alarm(labels: np.ndarray):
    """Computes prediction and utility for the single optimal alarm per patient."""
    preds = np.zeros(len(labels), dtype=int)
    if labels.max() == 1:
        t_onset = int(np.argmax(labels))
        opt_t = max(0, t_onset - 6)
        preds[opt_t] = 1
    achieved, best = calculate_patient_utility(labels, preds)
    return preds, achieved, best


def calculate_best_persistent_alarm(labels: np.ndarray):
    """Computes prediction and utility when issuing persistent alarms from optimal onset window onwards."""
    preds = np.zeros(len(labels), dtype=int)
    if labels.max() == 1:
        t_onset = int(np.argmax(labels))
        opt_t = max(0, t_onset - 6)
        preds[opt_t:] = 1
    achieved, best = calculate_patient_utility(labels, preds)
    return preds, achieved, best


def calculate_cohort_utility(all_labels, all_preds):
    """Computes total cohort achieved, total best, and normalized utility score."""
    tot_ach, tot_best = 0.0, 0.0
    for lbls, prs in zip(all_labels, all_preds):
        a, b = calculate_patient_utility(lbls, prs)
        tot_ach += a
        tot_best += b
    norm = tot_ach / tot_best if tot_best > 0 else 0.0
    return tot_ach, tot_best, norm


def calculate_patient_adaptive_threshold_ceiling(all_labels, all_probs, cooldown_hours=72):
    """
    Corrected Score-Based Patient-Adaptive Threshold Ceiling.
    For each patient independently:
      - Considers every candidate threshold t derived from y_prob.
      - An alarm occurs ONLY when y_prob >= t at the FIRST crossing hour (with optional cooldown).
      - Finds the threshold t_i* (and resulting alarm hour) that maximizes THIS PATIENT'S utility.
      - Non-septic patients choose a threshold above max(y_prob) -> 'never alarm' (0.0 utility).
      - Septic patients test all candidate thresholds that trigger an alarm at a REAL probability crossing.
    Returns: (normalized_ceiling, total_achieved, total_best, patient_rows_df)
    """
    patient_rows = []
    tot_achieved = 0.0
    tot_best = 0.0

    for idx, (lbls, prs) in enumerate(zip(all_labels, all_probs)):
        lbls = np.asarray(lbls, dtype=int)
        prs = np.asarray(prs, dtype=float)
        T = len(lbls)
        is_sepsis = int(lbls.max()) == 1

        if not is_sepsis:
            # Non-septic patient: optimal threshold is above max(prs) -> never alarm -> 0.0 utility
            best_ach = 0.0
            best_th = 1.01
            best_alarm_hour = -1
            prob_at_alarm = 0.0
            best_possible = 0.0
        else:
            best_possible = 1.0
            t_onset = int(np.argmax(lbls))

            # Default: never alarm (missed sepsis = -2.0)
            best_ach, _ = calculate_patient_utility(lbls, np.zeros(T, dtype=int))
            best_th = 1.01
            best_alarm_hour = -1
            prob_at_alarm = 0.0

            # Candidate thresholds: all unique non-zero probabilities in prs, plus grid values
            candidate_thresholds = np.unique(np.concatenate([prs, np.linspace(0.001, 0.999, 200)]))
            candidate_thresholds = np.sort(candidate_thresholds)[::-1]  # high to low

            for th in candidate_thresholds:
                alarm_indices = np.where(prs >= th)[0]
                if len(alarm_indices) == 0:
                    continue

                t_first = alarm_indices[0]
                preds = np.zeros(T, dtype=int)
                
                if cooldown_hours is None or cooldown_hours == 0:
                    preds[t_first] = 1
                else:
                    # Apply cooldown alert suppression from t_first
                    curr_t = t_first
                    while curr_t < T:
                        if prs[curr_t] >= th:
                            preds[curr_t] = 1
                            curr_t += cooldown_hours
                        else:
                            curr_t += 1

                ach, _ = calculate_patient_utility(lbls, preds)
                if ach > best_ach:
                    best_ach = ach
                    best_th = float(th)
                    best_alarm_hour = int(t_first)
                    prob_at_alarm = float(prs[t_first])

        tot_achieved += best_ach
        tot_best += best_possible

        patient_rows.append({
            "patient_id": idx,
            "is_sepsis": int(is_sepsis),
            "length_hours": T,
            "onset_hour": int(np.argmax(lbls)) if is_sepsis else -1,
            "optimal_threshold": best_th,
            "first_alarm_hour": best_alarm_hour,
            "prob_at_alarm": prob_at_alarm,
            "optimal_utility_contribution": best_ach,
            "best_possible_utility": best_possible
        })

    norm_ceiling = tot_achieved / tot_best if tot_best > 0 else 0.0
    return norm_ceiling, tot_achieved, tot_best, pd.DataFrame(patient_rows)


