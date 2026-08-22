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
