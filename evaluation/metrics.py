"""
metrics.py
----------
Secondary evaluation metrics:
  - AUROC, AUPRC
  - F1, Precision, Recall at optimal threshold
  - Expected Calibration Error (ECE) for the uncertainty module
  - Timing analysis: mean hours-before-onset for true positives
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)


# ---------------------------------------------------------------------------
# Standard classification metrics
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute AUROC, AUPRC, F1, Precision, Recall at a given threshold.

    Parameters
    ----------
    y_true   : 1D array of ground-truth binary labels {0, 1}
    y_proba  : 1D array of predicted probabilities [0, 1]
    threshold: decision threshold for binary predictions

    Returns
    -------
    dict of metric_name → value
    """
    y_pred = (y_proba >= threshold).astype(int)

    results = {
        "auroc":     float(roc_auc_score(y_true, y_proba)),
        "auprc":     float(average_precision_score(y_true, y_proba)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": threshold,
    }
    return results


# ---------------------------------------------------------------------------
# Timing analysis
# ---------------------------------------------------------------------------

def compute_timing_analysis(
    all_labels:      List[np.ndarray],
    all_predictions: List[np.ndarray],
    dt_early:        int = 12,  # max valid hours before onset
    dt_late:         int = 3,   # max valid hours after onset
) -> dict:
    """
    For true-positive sepsis patients (alarm raised within valid clinical window),
    compute statistics on how many hours BEFORE onset the first valid alarm was raised.

    Alarms > 12h before onset are premature non-actionable alarms (outside utility window).
    Alarms > 3h after onset are late detections.

    Negative value = alarm raised AFTER onset (late detection).
    Positive value = alarm raised BEFORE onset (early detection).
    """
    lead_times = []

    for labels, preds in zip(all_labels, all_predictions):
        labels = np.asarray(labels, dtype=int)
        preds  = np.asarray(preds,  dtype=int)

        is_sepsis = labels.max() == 1
        if not is_sepsis:
            continue

        t_onset = int(np.argmax(labels))
        alarm_times = np.where(preds == 1)[0]

        if len(alarm_times) == 0:
            continue  # false negative — no alarm

        # Filter for alarms within the valid utility evaluation window [t_onset - 12, t_onset + 3]
        valid_alarms = [t for t in alarm_times if (t_onset - dt_early) <= t <= (t_onset + dt_late)]

        if not valid_alarms:
            # First alarm was either > 12h premature or > 3h late
            # If there's any alarm prior to t_onset + dt_late, take the first one
            first_alarm = alarm_times[0]
            if first_alarm <= t_onset + dt_late:
                # Cap premature lead time at dt_early (12h) for reporting
                lead = min(float(dt_early), float(t_onset - first_alarm))
                lead_times.append(lead)
            continue

        t_alarm = valid_alarms[0]
        lead_times.append(float(t_onset - t_alarm))   # positive = early

    if not lead_times:
        return {"n_tp": 0, "mean_lead_h": None, "median_lead_h": None}

    lt = np.array(lead_times)
    return {
        "n_tp":           len(lt),
        "mean_lead_h":    float(lt.mean()),
        "median_lead_h":  float(np.median(lt)),
        "pct_early_6h":   float((lt >= 6).mean() * 100),   # % caught ≥6h early
        "pct_early_1h":   float((lt >= 1).mean() * 100),   # % caught ≥1h early
        "pct_late":       float((lt < 0).mean() * 100),    # % caught after onset
    }



# ---------------------------------------------------------------------------
# Expected Calibration Error (ECE)
# ---------------------------------------------------------------------------

def compute_ece(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error.
    ECE = Σ_b  (|B_b| / N) × |acc(B_b) - conf(B_b)|

    A well-calibrated model has ECE ≈ 0.
    """
    y_true  = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)
    N = len(y_true)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0

    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        mask = (y_proba >= low) & (y_proba < high)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_proba[mask].mean()
        ece += (mask.sum() / N) * abs(acc - conf)

    return float(ece)


import matplotlib.pyplot as plt

def plot_reliability_diagram(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str] = None
):
    """
    Plots a Reliability Diagram (Calibration Curve) for binary predictions.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)
    
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    
    true_props = []
    pred_props = []
    
    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        mask = (y_proba >= low) & (y_proba < high)
        if mask.sum() > 0:
            true_props.append(y_true[mask].mean())
            pred_props.append(y_proba[mask].mean())
        else:
            true_props.append(np.nan)
            pred_props.append(np.nan)
            
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k:', label="Perfect Calibration")
    plt.plot(pred_props, true_props, marker='o', label="Model Calibration", color='blue')
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of True Positives")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()


# ---------------------------------------------------------------------------
# Full evaluation report
# ---------------------------------------------------------------------------

def full_evaluation_report(
    all_labels:       List[np.ndarray],
    all_proba:        List[np.ndarray],
    all_predictions:  List[np.ndarray],
    utility_score:    Optional[float] = None,
    split_name:       str = "test",
) -> dict:
    """
    Compile a complete evaluation report and pretty-print it.
    """
    y_true_flat  = np.concatenate(all_labels)
    y_proba_flat = np.concatenate(all_proba)
    y_pred_flat  = np.concatenate(all_predictions)

    clf_metrics = compute_classification_metrics(y_true_flat, y_proba_flat)
    timing      = compute_timing_analysis(all_labels, all_predictions)
    ece         = compute_ece(y_true_flat, y_proba_flat)

    report = {
        "split": split_name,
        **clf_metrics,
        "ece":   ece,
        **{f"timing_{k}": v for k, v in timing.items()},
    }
    if utility_score is not None:
        report["utility_score"] = utility_score

    print(f"\n{'=' * 55}")
    print(f"  EVALUATION REPORT — {split_name.upper()}")
    print(f"{'=' * 55}")
    print(f"  Utility Score  : {utility_score:.4f}" if utility_score else "  Utility Score  : N/A")
    print(f"  AUROC          : {clf_metrics['auroc']:.4f}")
    print(f"  AUPRC          : {clf_metrics['auprc']:.4f}")
    print(f"  F1             : {clf_metrics['f1']:.4f}")
    print(f"  Precision      : {clf_metrics['precision']:.4f}")
    print(f"  Recall         : {clf_metrics['recall']:.4f}")
    print(f"  ECE            : {ece:.4f}")
    if timing.get("n_tp"):
        print(f"  -- Timing (true positives) --")
        print(f"  N true positives   : {timing['n_tp']}")
        print(f"  Mean lead time     : {timing['mean_lead_h']:.1f} h")
        print(f"  Caught >=6h early   : {timing['pct_early_6h']:.1f}%")
        print(f"  Caught >=1h early   : {timing['pct_early_1h']:.1f}%")
        print(f"  Late detections    : {timing['pct_late']:.1f}%")
    print(f"{'=' * 55}\n")

    return report
