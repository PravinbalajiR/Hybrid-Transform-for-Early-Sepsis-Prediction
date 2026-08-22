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


def calculate_per_patient_optimal_hindsight(all_labels, all_probs):
    """
    Computes PER_PATIENT_OPTIMAL_HINDSIGHT_CEILING across all patients independently.
    For each patient, computes the highest achievable utility contribution without imposing
    any global threshold or global cooldown policy across patients.
    Returns: (normalized_ceiling, total_achieved, total_best, patient_rows)
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
            # Non-septic patient: optimal independent choice is never alarm -> 0.0 utility
            best_p_ach = 0.0
            best_p_hour = -1
            best_p_best = 0.0
        else:
            # Septic patient: search over all possible single alarm hours t in {0, ..., T-1} plus no alarm
            best_p_ach = -2.0  # default no alarm
            best_p_hour = -1
            best_p_best = 1.0
            t_onset = int(np.argmax(lbls))

            # 1. No alarm
            ach_no, _ = calculate_patient_utility(lbls, np.zeros(T, dtype=int))
            if ach_no > best_p_ach:
                best_p_ach = ach_no
                best_p_hour = -1

            # 2. Test every single hour t as the first alarm time
            for t_al in range(T):
                p_test = np.zeros(T, dtype=int)
                p_test[t_al] = 1
                ach_t, _ = calculate_patient_utility(lbls, p_test)
                if ach_t > best_p_ach:
                    best_p_ach = ach_t
                    best_p_hour = t_al

        tot_achieved += best_p_ach
        tot_best += best_p_best

        patient_rows.append({
            "patient_id": idx,
            "is_sepsis": int(is_sepsis),
            "length_hours": T,
            "onset_hour": int(np.argmax(lbls)) if is_sepsis else -1,
            "optimal_hour": best_p_hour,
            "optimal_utility_contribution": best_p_ach,
            "best_possible_utility": best_p_best
        })

    norm_ceiling = tot_achieved / tot_best if tot_best > 0 else 0.0
    return norm_ceiling, tot_achieved, tot_best, pd.DataFrame(patient_rows)

