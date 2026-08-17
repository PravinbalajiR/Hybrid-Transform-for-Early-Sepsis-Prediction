"""
run_m3_tap_phase3_policy_search.py
----------------------------------
M3-TAP Phase 3A: Validation-Only Comprehensive Policy Search.
Evaluates thousands of candidate temporal alert policies (cooldown, persistence, smoothing,
hysteresis, alert caps, adaptive cooldown, composites) strictly on Validation cohort (N=2,034).
Zero test leakage.
"""

import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from evaluation.utility_score import compute_utility_score
from evaluation.metrics import compute_timing_analysis
from scripts.temporal_alert_policy import (
    BaseAlertPolicy,
    NaiveThresholdPolicy,
    PersistencePolicy,
    HysteresisPolicy,
    CooldownPolicy,
    MovingAveragePolicy,
    ExponentialMovingAveragePolicy,
    CombinedTAPPolicy,
)

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def print_flush(msg: str):
    print(msg, flush=True)

# --------------------------------------------------------------------------------------
# EXTENDED POLICY CLASSES FOR PHASE 3A
# --------------------------------------------------------------------------------------

class AlertCapPolicy(BaseAlertPolicy):
    def __init__(self, base_policy: BaseAlertPolicy, max_alerts_per_24h: int = 1):
        super().__init__(f"Cap{max_alerts_per_24h}/24h({base_policy.name})")
        self.base_policy = base_policy
        self.max_alerts = max_alerts_per_24h

    def generate_alerts_for_patient(self, probs: np.ndarray) -> np.ndarray:
        raw_alerts = self.base_policy.generate_alerts_for_patient(probs)
        T = len(raw_alerts)
        if T == 0 or self.max_alerts <= 0: return raw_alerts
        
        alerts = np.zeros(T, dtype=int)
        for t in range(T):
            if raw_alerts[t] == 1:
                start_window = max(0, t - 23)
                window_alerts = alerts[start_window:t].sum()
                if window_alerts < self.max_alerts:
                    alerts[t] = 1
        return alerts


class MedianSmoothingPolicy(BaseAlertPolicy):
    def __init__(self, threshold: float, window_W: int = 3, cooldown_hours: int = 24):
        super().__init__(f"MedianSmooth(th={threshold:.2f}, W={window_W}h, C={cooldown_hours}h)")
        self.threshold = threshold
        self.window_W = window_W
        self.cooldown_hours = cooldown_hours

    def generate_alerts_for_patient(self, probs: np.ndarray) -> np.ndarray:
        T = len(probs)
        if T == 0: return np.zeros(0, dtype=int)
        
        smooth_p = np.zeros(T)
        for t in range(T):
            start = max(0, t - self.window_W + 1)
            smooth_p[t] = np.median(probs[start : t + 1])
            
        alerts = np.zeros(T, dtype=int)
        cooldown_rem = 0
        for t in range(T):
            if cooldown_rem > 0:
                cooldown_rem -= 1
                continue
            if smooth_p[t] >= self.threshold:
                alerts[t] = 1
                if self.cooldown_hours > 0:
                    cooldown_rem = self.cooldown_hours
        return alerts


class RiskAdaptiveCooldownPolicy(BaseAlertPolicy):
    def __init__(self, threshold: float, c_low_risk: int = 24, c_high_risk: int = 6, high_risk_cutoff: float = 0.40):
        super().__init__(f"RiskAdaptive(th={threshold:.2f}, C_low={c_low_risk}h, C_high={c_high_risk}h)")
        self.threshold = threshold
        self.c_low_risk = max(0, c_low_risk)
        self.c_high_risk = max(0, c_high_risk)
        self.high_risk_cutoff = high_risk_cutoff

    def generate_alerts_for_patient(self, probs: np.ndarray) -> np.ndarray:
        T = len(probs)
        if T == 0: return np.zeros(0, dtype=int)
        alerts = np.zeros(T, dtype=int)
        cooldown_rem = 0
        for t in range(T):
            if cooldown_rem > 0:
                cooldown_rem -= 1
                continue
            p_t = probs[t]
            if p_t >= self.threshold:
                alerts[t] = 1
                if p_t >= self.high_risk_cutoff:
                    cooldown_rem = self.c_high_risk
                else:
                    cooldown_rem = self.c_low_risk
        return alerts


class PersistenceCooldownPolicy(BaseAlertPolicy):
    def __init__(self, threshold: float, K_persist: int, cooldown_hours: int):
        super().__init__(f"PersistCooldown(th={threshold:.2f}, K={K_persist}, C={cooldown_hours}h)")
        self.threshold = threshold
        self.K_persist = max(1, K_persist)
        self.cooldown_hours = max(0, cooldown_hours)

    def generate_alerts_for_patient(self, probs: np.ndarray) -> np.ndarray:
        T = len(probs)
        if T == 0: return np.zeros(0, dtype=int)
        alerts = np.zeros(T, dtype=int)
        cooldown_rem = 0
        consecutive = 0
        for t in range(T):
            if cooldown_rem > 0:
                cooldown_rem -= 1
                consecutive = 0
                continue
            if probs[t] >= self.threshold:
                consecutive += 1
                if consecutive >= self.K_persist:
                    alerts[t] = 1
                    if self.cooldown_hours > 0:
                        cooldown_rem = self.cooldown_hours
                    consecutive = 0
            else:
                consecutive = 0
        return alerts


class HysteresisCooldownPolicy(BaseAlertPolicy):
    def __init__(self, th_high: float, th_low: float, cooldown_hours: int):
        super().__init__(f"HysteresisCooldown(high={th_high:.2f}, low={th_low:.2f}, C={cooldown_hours}h)")
        self.th_high = th_high
        self.th_low = min(th_high, th_low)
        self.cooldown_hours = max(0, cooldown_hours)

    def generate_alerts_for_patient(self, probs: np.ndarray) -> np.ndarray:
        T = len(probs)
        if T == 0: return np.zeros(0, dtype=int)
        alerts = np.zeros(T, dtype=int)
        active = False
        cooldown_rem = 0
        for t in range(T):
            if cooldown_rem > 0:
                cooldown_rem -= 1
                active = False
                continue
            p_t = probs[t]
            if not active:
                if p_t >= self.th_high:
                    active = True
                    alerts[t] = 1
            else:
                if p_t >= self.th_low:
                    alerts[t] = 1
                else:
                    active = False
                    if self.cooldown_hours > 0:
                        cooldown_rem = self.cooldown_hours
        return alerts


# --------------------------------------------------------------------------------------
# COHORT METRICS COMPUTATION
# --------------------------------------------------------------------------------------

def evaluate_policy_on_validation(policy, val_labels, val_probs, category_name: str = "General"):
    val_preds = policy.generate_alerts_cohort(val_probs)
    u_norm = compute_utility_score(val_labels, val_preds)

    y_true_flat = np.concatenate(val_labels)
    y_pred_flat = np.concatenate(val_preds)

    tp_h = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
    fp_h = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
    fn_h = np.sum((y_true_flat == 1) & (y_pred_flat == 0))
    tn_h = np.sum((y_true_flat == 0) & (y_pred_flat == 0))

    prec = tp_h / (tp_h + fp_h) if (tp_h + fp_h) > 0 else 0.0
    rec = tp_h / (tp_h + fn_h) if (tp_h + fn_h) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp_h / (fp_h + tn_h) if (fp_h + tn_h) > 0 else 0.0

    timing = compute_timing_analysis(val_labels, val_preds)

    n_sepsis = 0
    n_tp_sepsis = 0
    total_alerts = int(y_pred_flat.sum())
    n_patients_alerted = 0

    for lbls, prs in zip(val_labels, val_preds):
        if prs.max() == 1:
            n_patients_alerted += 1
        if lbls.max() == 1:
            n_sepsis += 1
            if prs.max() == 1:
                n_tp_sepsis += 1

    patient_detection_rate = n_tp_sepsis / n_sepsis if n_sepsis > 0 else 0.0
    alerts_per_patient = total_alerts / len(val_labels)

    return {
        "category": category_name,
        "policy_name": policy.name,
        "utility": float(u_norm),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "fpr_h": float(fpr),
        "patient_detection_rate": float(patient_detection_rate),
        "n_tp_patients": n_tp_sepsis,
        "n_fn_patients": n_sepsis - n_tp_sepsis,
        "n_sepsis_patients": n_sepsis,
        "mean_lead_h": timing.get("mean_lead_h", 0.0) if timing.get("mean_lead_h") is not None else 0.0,
        "median_lead_h": timing.get("median_lead_h", 0.0) if timing.get("median_lead_h") is not None else 0.0,
        "pct_early_1h": timing.get("pct_early_1h", 0.0) if timing.get("pct_early_1h") is not None else 0.0,
        "pct_early_6h": timing.get("pct_early_6h", 0.0) if timing.get("pct_early_6h") is not None else 0.0,
        "pct_early_12h": timing.get("pct_early_12h", 0.0) if timing.get("pct_early_12h") is not None else 0.0,
        "total_alerts": total_alerts,
        "alerts_per_patient": alerts_per_patient,
        "n_patients_alerted": n_patients_alerted,
        "policy_obj": policy,
    }


def main():
    print_flush("=" * 95)
    print_flush("   M3-TAP PHASE 3A: COMPREHENSIVE VALIDATION-ONLY POLICY SEARCH (N=2,034)")
    print_flush("=" * 95)

    val_npz_path = RESULTS_DIR / "m3_final_val_predictions.npz"
    if not val_npz_path.exists():
        print_flush("Error: Validation predictions NPZ missing in results/!")
        sys.exit(1)

    val_data = np.load(val_npz_path, allow_pickle=True)
    val_y_true, val_y_prob, val_lens = val_data["y_true_flat"], val_data["y_proba_flat"], val_data["patient_lengths"]
    val_labels, val_probs = [], []
    curr = 0
    for l in val_lens:
        val_labels.append(val_y_true[curr : curr + l])
        val_probs.append(val_y_prob[curr : curr + l])
        curr += l

    print_flush(f"Loaded Validation Cohort : {len(val_labels):,} patients ({len(val_y_true):,} hourly records).\n")

    candidate_policies = []

    # 1. Fine Threshold x Cooldown Sweep (th=0.05..0.50 step 0.01, C=2..48h)
    for th in np.arange(0.05, 0.51, 0.01):
        for C in [2, 4, 6, 8, 12, 18, 24, 36, 48]:
            candidate_policies.append(("1. ThresholdxCooldown", CooldownPolicy(float(th), int(C))))

    # 2. Persistence Requirement (k=1..4, th=0.05..0.50 step 0.02, C=6..36h)
    for th in np.arange(0.05, 0.51, 0.02):
        for K in [1, 2, 3, 4]:
            for C in [6, 12, 24, 36]:
                candidate_policies.append(("2. Persistence", PersistenceCooldownPolicy(float(th), int(K), int(C))))

    # 3. Temporal Smoothing (rolling mean 2,3,6h & rolling median 3h)
    for th in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
        for C in [12, 24, 36]:
            for W in [2, 3, 6]:
                candidate_policies.append(("3. TemporalSmoothingMean", CombinedTAPPolicy(float(th), float(th), 1, int(C), int(W), 1.0)))
            candidate_policies.append(("3. TemporalSmoothingMedian", MedianSmoothingPolicy(float(th), 3, int(C))))

    # 4. Hysteresis (alert_on high vs alert_off low)
    for high in np.arange(0.15, 0.51, 0.05):
        for low in [0.05, 0.10, 0.15, 0.20]:
            if low < high:
                for C in [12, 24, 36]:
                    candidate_policies.append(("4. Hysteresis", HysteresisCooldownPolicy(float(high), float(low), int(C))))

    # 5. Alert Cap (max 1, 2, 3 per 24h)
    for th in [0.15, 0.20, 0.25]:
        for cap in [1, 2, 3]:
            base_pol = CooldownPolicy(float(th), 12)
            candidate_policies.append(("5. AlertCap", AlertCapPolicy(base_pol, int(cap))))

    # 6. Adaptive Cooldown
    for th in [0.10, 0.15, 0.20, 0.25]:
        for c_low in [24, 36]:
            for c_high in [6, 12]:
                candidate_policies.append(("6. RiskAdaptiveCooldown", RiskAdaptiveCooldownPolicy(float(th), int(c_low), int(c_high), 0.40)))

    total_candidates = len(candidate_policies)
    print_flush(f"Generated {total_candidates:,} candidate validation policies.")

    val_sweep_records = []
    best_val_u_so_far = -999.0

    for idx, (cat_name, pol) in enumerate(candidate_policies):
        res = evaluate_policy_on_validation(pol, val_labels, val_probs, cat_name)
        val_sweep_records.append(res)
        if res['utility'] > best_val_u_so_far:
            best_val_u_so_far = res['utility']
            print_flush(f"   [NEW BEST VAL UTILITY] {res['utility']:+.6f} | Category: {cat_name} | Policy: {pol.name}")

    df_val_matrix = pd.DataFrame(val_sweep_records)
    df_val_matrix_clean = df_val_matrix.drop(columns=["policy_obj"])
    df_val_matrix_clean.to_csv(RESULTS_DIR / "m3_tap_phase3_policy_sweep.csv", index=False)

    print_flush(f"\nSaved full Phase 3A validation policy sweep ({len(df_val_matrix_clean):,} policies) to: {RESULTS_DIR / 'm3_tap_phase3_policy_sweep.csv'}")

if __name__ == "__main__":
    main()
