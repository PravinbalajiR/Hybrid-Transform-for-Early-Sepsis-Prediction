"""
temporal_alert_policy.py
------------------------
Temporal Alert Policy Engine for M3 Advancement (M3-TAP).
Decouples continuous risk estimation p_t from discrete alert decision generation A_t.

Provides policies:
1. Base Naive Policy (p_t >= threshold)
2. Persistence Policy (p_t >= threshold for K consecutive hours)
3. Hysteresis Policy (activate at th_on, deactivate at th_off)
4. Cooldown Policy (suppress new alerts for C hours after activation)
5. Moving Average Policy (SMA over window K)
6. Exponential Moving Average Policy (EMA with alpha)
7. Combined M3-TAP Policy (Smoothing + Persistence + Hysteresis + Cooldown)
"""

import numpy as np
from typing import List, Dict, Any

class BaseAlertPolicy:
    def __init__(self, name: str):
        self.name = name

    def generate_alerts_for_patient(self, probs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def generate_alerts_cohort(self, all_probs: List[np.ndarray]) -> List[np.ndarray]:
        return [self.generate_alerts_for_patient(p) for p in all_probs]


class NaiveThresholdPolicy(BaseAlertPolicy):
    def __init__(self, threshold: float):
        super().__init__(f"Naive(th={threshold:.2f})")
        self.threshold = threshold

    def generate_alerts_for_patient(self, probs: np.ndarray) -> np.ndarray:
        return (probs >= self.threshold).astype(int)


class PersistencePolicy(BaseAlertPolicy):
    def __init__(self, threshold: float, K: int):
        super().__init__(f"Persistence(th={threshold:.2f}, K={K})")
        self.threshold = threshold
        self.K = max(1, K)

    def generate_alerts_for_patient(self, probs: np.ndarray) -> np.ndarray:
        T = len(probs)
        alerts = np.zeros(T, dtype=int)
        consecutive = 0
        for t in range(T):
            if probs[t] >= self.threshold:
                consecutive += 1
                if consecutive >= self.K:
                    alerts[t] = 1
            else:
                consecutive = 0
        return alerts


class HysteresisPolicy(BaseAlertPolicy):
    def __init__(self, th_on: float, th_off: float):
        super().__init__(f"Hysteresis(th_on={th_on:.2f}, th_off={th_off:.2f})")
        self.th_on = th_on
        self.th_off = min(th_on, th_off)

    def generate_alerts_for_patient(self, probs: np.ndarray) -> np.ndarray:
        T = len(probs)
        alerts = np.zeros(T, dtype=int)
        active = False
        for t in range(T):
            if not active:
                if probs[t] >= self.th_on:
                    active = True
                    alerts[t] = 1
            else:
                if probs[t] >= self.th_off:
                    alerts[t] = 1
                else:
                    active = False
        return alerts


class CooldownPolicy(BaseAlertPolicy):
    def __init__(self, threshold: float, cooldown_hours: int):
        super().__init__(f"Cooldown(th={threshold:.2f}, C={cooldown_hours}h)")
        self.threshold = threshold
        self.cooldown_hours = max(0, cooldown_hours)

    def generate_alerts_for_patient(self, probs: np.ndarray) -> np.ndarray:
        T = len(probs)
        alerts = np.zeros(T, dtype=int)
        cooldown_remaining = 0
        for t in range(T):
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
                continue
            if probs[t] >= self.threshold:
                alerts[t] = 1
                cooldown_remaining = self.cooldown_hours
        return alerts


class MovingAveragePolicy(BaseAlertPolicy):
    def __init__(self, threshold: float, window_K: int):
        super().__init__(f"SMA(th={threshold:.2f}, K={window_K})")
        self.threshold = threshold
        self.window_K = max(1, window_K)

    def generate_alerts_for_patient(self, probs: np.ndarray) -> np.ndarray:
        T = len(probs)
        smoothed = np.zeros(T)
        for t in range(T):
            start_idx = max(0, t - self.window_K + 1)
            smoothed[t] = probs[start_idx : t + 1].mean()
        return (smoothed >= self.threshold).astype(int)


class ExponentialMovingAveragePolicy(BaseAlertPolicy):
    def __init__(self, threshold: float, alpha: float):
        super().__init__(f"EMA(th={threshold:.2f}, alpha={alpha:.2f})")
        self.threshold = threshold
        self.alpha = float(np.clip(alpha, 0.01, 1.0))

    def generate_alerts_for_patient(self, probs: np.ndarray) -> np.ndarray:
        T = len(probs)
        ema = np.zeros(T)
        r_prev = probs[0] if T > 0 else 0.0
        for t in range(T):
            r_curr = self.alpha * probs[t] + (1.0 - self.alpha) * r_prev
            ema[t] = r_curr
            r_prev = r_curr
        return (ema >= self.threshold).astype(int)


class CombinedTAPPolicy(BaseAlertPolicy):
    def __init__(
        self,
        th_on: float,
        th_off: float,
        K_persist: int = 1,
        cooldown_hours: int = 0,
        sma_window: int = 1,
        ema_alpha: float = 1.0,
    ):
        name = f"M3-TAP(on={th_on:.2f}, off={th_off:.2f}, K={K_persist}, C={cooldown_hours}h, W={sma_window}, alpha={ema_alpha:.2f})"
        super().__init__(name)
        self.th_on = th_on
        self.th_off = min(th_on, th_off)
        self.K_persist = max(1, K_persist)
        self.cooldown_hours = max(0, cooldown_hours)
        self.sma_window = max(1, sma_window)
        self.ema_alpha = float(np.clip(ema_alpha, 0.01, 1.0))

    def generate_alerts_for_patient(self, probs: np.ndarray) -> np.ndarray:
        T = len(probs)
        if T == 0:
            return np.zeros(0, dtype=int)

        # 1. Apply EMA / SMA smoothing
        smoothed = np.zeros(T)
        r_prev = probs[0]
        for t in range(T):
            # Combined EMA + SMA window
            start_idx = max(0, t - self.sma_window + 1)
            sma_val = probs[start_idx : t + 1].mean()
            r_curr = self.ema_alpha * sma_val + (1.0 - self.ema_alpha) * r_prev
            smoothed[t] = r_curr
            r_prev = r_curr

        # 2. Apply Persistence + Hysteresis + Cooldown
        alerts = np.zeros(T, dtype=int)
        active = False
        consecutive_high = 0
        cooldown_remaining = 0

        for t in range(T):
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
                active = False
                consecutive_high = 0
                continue

            r_t = smoothed[t]

            if not active:
                if r_t >= self.th_on:
                    consecutive_high += 1
                    if consecutive_high >= self.K_persist:
                        active = True
                        alerts[t] = 1
                else:
                    consecutive_high = 0
            else:
                if r_t >= self.th_off:
                    alerts[t] = 1
                else:
                    active = False
                    consecutive_high = 0
                    if self.cooldown_hours > 0:
                        cooldown_remaining = self.cooldown_hours

        return alerts
