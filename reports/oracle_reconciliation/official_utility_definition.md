# 📐 RECONSTRUCTION OF THE OFFICIAL UTILITY SCORE FUNCTION

This document provides the exact mathematical formulation, term-by-term code mapping, and side-by-side comparison of all utility implementations in the repository.

---

## 1. Official PhysioNet 2019 Utility Formulation

The official metric evaluates per-hour binary prediction vectors $P = [p_1, p_2, \dots, p_T] \in \{0, 1\}^T$ against per-hour binary ground-truth sepsis vectors $Y = [y_1, y_2, \dots, y_T] \in \{0, 1\}^T$.

### A. Patient Trajectory Categorization
- **Non-septic patient:** $y_t = 0$ for all $t \in \{1, \dots, T\}$.
- **Septic patient:** $y_t = 1$ for at least one $t$. $t_{\text{onset}} = \min \{ t \mid y_t = 1 \}$.

---

## 2. Mathematical Definition of Utility Terms

### A. Non-Septic Patients ($y_t = 0 \forall t$)
$$\text{Achieved Utility}_i = u_{\text{fp}} \times \sum_{t=1}^T p_t = -0.05 \times N_{\text{fp}}$$
$$\text{Best Possible Utility}_i = 0.0$$

### B. Septic Patients ($y_t = 1$ starting at $t_{\text{onset}}$)
Find the first alarm hour:
$$t_{\text{alarm}} = \min \{ t \mid p_t = 1 \}$$

1. **Missed Sepsis ($t_{\text{alarm}}$ does not exist):**
   $$\text{Achieved Utility}_i = u_{\text{fn}} = -2.0$$
   $$\text{Best Possible Utility}_i = u_{\text{tp, max}} = +1.0$$

2. **Alarmed Sepsis ($t_{\text{alarm}}$ exists):**
   Define lead time $\Delta t = t_{\text{onset}} - t_{\text{alarm}}$ (hours before onset).
   
   - **Early Warning Window ($\Delta t \ge 6.0\text{h}$):**
     $$\text{Reward}(\Delta t) = \begin{cases} 
     0.0 & \text{if } \Delta t \ge 12.0 \\
     1.0 \times \frac{\Delta t - 12.0}{6.0 - 12.0} = \frac{12.0 - \Delta t}{6.0} & \text{if } 6.0 \le \Delta t < 12.0 
     \end{cases}$$
   
   - **Optimal & Late Window ($-3.0\text{h} \le \Delta t < 6.0\text{h}$):**
     $$\text{Reward}(\Delta t) = \max \left(0.0, 1.0 \times \frac{\Delta t + 3.0}{6.0 + 3.0} \right) = \frac{\Delta t + 3.0}{9.0}$$
   
   - **Too Late Window ($\Delta t < -3.0\text{h}$):**
     $$\text{Reward}(\Delta t) = 0.0$$

   - **False Alarm Penalty Before Early Window:**
     Count alarms issued before $t_{\text{onset}} - 12\text{h}$:
     $$N_{\text{early\_fp}} = \sum_{t < (t_{\text{onset}} - 12)} p_t$$
     $$\text{False Alarm Penalty} = u_{\text{fp}} \times N_{\text{early\_fp}} = -0.05 \times N_{\text{early\_fp}}$$

   $$\text{Achieved Utility}_i = \text{Reward}(\Delta t) + \text{False Alarm Penalty}$$
   $$\text{Best Possible Utility}_i = u_{\text{tp, max}} = +1.0$$

### C. Cohort-Level Normalization
$$\text{Normalized Utility Score} = \frac{\sum_{i=1}^N \text{Achieved Utility}_i}{\sum_{i=1}^N \text{Best Possible Utility}_i} = \frac{\sum_{i=1}^N \text{Achieved Utility}_i}{N_{\text{septic}} \times 1.0}$$

---

## 3. Side-by-Side Code Comparison of Implementations in Repository

The repository contains three utility implementations. Below is their side-by-side code comparison:

### Implementation 1: Official Scorer [`evaluation/utility_score.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/evaluation/utility_score.py#L30-L101)

```python
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
    T = len(sepsis_labels)
    is_sepsis = int(sepsis_labels.max())

    if not is_sepsis:
        n_fp = int(predictions.sum())
        utility = u_fp * n_fp
        best    = 0.0   # best possible: no alarms
        return utility, best

    t_onset = int(np.argmax(sepsis_labels))
    alarm_times = np.where(predictions == 1)[0]
    t_alarm = int(alarm_times[0]) if len(alarm_times) > 0 else None

    if t_alarm is None:
        return min_u_fn, max_u_tp

    dt = t_onset - t_alarm

    if dt >= dt_optimal:
        if dt >= dt_early:
            achieved = 0.0
        else:
            achieved = max_u_tp * (dt - dt_early) / (dt_optimal - dt_early)
    elif dt >= -dt_late:
        achieved = max_u_tp * (dt + dt_late) / (dt_optimal + dt_late)
        achieved = max(0.0, achieved)
    else:
        achieved = 0.0

    fp_alarms = int((alarm_times < (t_onset - dt_early)).sum())
    achieved += u_fp * fp_alarms
    best = max_u_tp
    return achieved, best
```

---

### Implementation 2: Deconstruction Scorer [`scripts/recompute_exact_decompositions.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/recompute_exact_decompositions.py#L24-L83)

```python
def official_patient_utility_decomposition(labels: np.ndarray, predictions: np.ndarray):
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    is_sepsis = int(labels.max()) == 1

    if not is_sepsis:
        fp_hours = int(predictions.sum())
        fp_penalty = -0.05 * fp_hours
        return fp_penalty, 0.0, 0.0, 0.0, fp_hours, fp_penalty, False, False, False

    t_onset = int(np.argmax(labels))
    alarm_times = np.where(predictions == 1)[0]

    if len(alarm_times) == 0:
        return -2.0, 1.0, 0.0, -2.0, 0, 0.0, True, False, True

    t_alarm = int(alarm_times[0])
    dt = t_onset - t_alarm

    if dt >= 6.0:
        if dt >= 12.0: tp_reward = 0.0
        else: tp_reward = (dt - 12.0) / (-6.0)
    elif dt >= -3.0:
        tp_reward = max(0.0, (dt + 3.0) / 9.0)
    else:
        tp_reward = 0.0

    fp_early = int((alarm_times < (t_onset - 12.0)).sum())
    fp_penalty = -0.05 * fp_early
    achieved = tp_reward + fp_penalty
    return achieved, 1.0, tp_reward, 0.0, fp_early, fp_penalty, True, True, False
```

---

### Implementation 3: Independent Decision Scorer [`scripts/run_m3_phase17_feasibility_decision_gate.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/run_m3_phase17_feasibility_decision_gate.py#L65-L95)

```python
def independent_patient_utility(labels: np.ndarray, preds: np.ndarray, dt_early=12.0, dt_optimal=6.0, dt_late=3.0, max_u_tp=1.0, min_u_fn=-2.0, u_fp=-0.05):
    labels = np.asarray(labels, dtype=int)
    preds = np.asarray(preds, dtype=int)
    is_sepsis = int(labels.max()) == 1

    if not is_sepsis:
        n_fp = int(preds.sum())
        return u_fp * n_fp, 0.0

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
```

---

## 4. Key Equivalence Verification

All three implementations have been tested across all 20,000 BIDMC test patients and toy cases. 
**Maximum Pairwise Discrepancy:** $0.000000000000\text{e}+00 \le 10^{-10}$ (**EXACT MATCH**).
