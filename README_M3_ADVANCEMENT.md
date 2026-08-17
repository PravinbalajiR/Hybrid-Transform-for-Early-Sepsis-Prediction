# README — M3 ADVANCEMENT: FROM HIGH DISCRIMINATION TO POSITIVE OPERATIONAL UTILITY

## Project
**Sepsis-Hybrid-Transformer — M3 Advancement**

## Current scientific status
The current M3 model has successfully passed the following audits:
- Held-out test cohort: **20,000 patients**
- Test hourly observations: **753,927**
- Septic patients: **1,066**
- Non-septic patients: **18,934**
- AUROC: **0.961663**
- AUPRC: **0.423062**
- ECE: **0.0407**
- Brier score: **0.0213**
- Official scorer equivalence: verified
- Validation utility-optimal threshold: **0.44**
- Test utility at validation-locked `th=0.44`: **−1.1440**
- Test utility at `th=0.60`: **−0.9535**
- Test utility at `th=0.78`: **−0.8696**

The negative utility result is **not to be hidden, altered, or selectively reported**.

The scientific objective of the next phase is to determine whether the M3 probability predictions can be transformed into a more clinically realistic temporal alert policy that reduces unnecessary alarms while preserving early detection.

---

# 1. PRIMARY RESEARCH QUESTION
The next experiment must answer:
> **Can the high-discrimination M3 risk representation be converted into positive PhysioNet operational utility through a validation-locked temporal alert-generation policy without modifying the held-out test predictions or leaking test information into policy selection?**

The central hypothesis is:
> M3 produces useful continuous risk estimates, but naive hourly thresholding creates excessive alert fragmentation and false-alarm burden. A temporally constrained alert policy may convert the continuous risk signal into fewer, more persistent, and better-timed alerts, improving official utility.

---

# 2. DO NOT REPLACE M3
The current M3 model is the baseline. Do **not** immediately redesign the Transformer. The first advancement must operate on the existing M3 probability predictions.

The experimental structure is:
```text
M3
 │
 ├── physiological values
 ├── missingness masks
 ├── temporal gaps Δt
 │
 ▼
Transformer
 │
 ▼
p(sepsis | history)
 │
 ▼
NEW TEMPORAL ALERT POLICY (M3-TAP)
 │
 ├── persistence
 ├── hysteresis
 ├── cooldown
 ├── temporal smoothing
 ├── risk accumulation
 └── optional uncertainty constraint
 │
 ▼
Alert sequence
 │
 ▼
Official PhysioNet scorer
 │
 ▼
Utility
```

---

# 3. SCIENTIFIC PRINCIPLE
Separate continuous risk estimation $p_t = P(Y_t=1 \mid X_{1:t})$ from decision alert generation $A_t = \pi(p_{t-K:t}, A_{t-1}, \text{history})$.

---

# 4. EXPERIMENTAL GATES
- **Gate A (No test-set policy optimization):** All policy parameters must be selected using the **validation set only**.
- **Gate B (Official scorer):** All final utility evaluations must use the verified official PhysioNet utility implementation.
- **Gate C (Patient-level splitting):** Zero patient overlap between splits.
- **Gate D (No threshold cherry-picking):** Validation selection $\to$ freeze policy $\to$ single-pass test evaluation.
- **Gate E (Complete reporting):** Report AUROC, AUPRC, ECE, Brier, hourly P/R/F1, patient detection/missed rates, non-sepsis FPR/h, all-hours alarm rate, lead time, $\ge$1h/$\ge$6h warning rates, official utility, and exact decomposition.

---

# 5. EXECUTION STAGES & SCRIPTS PROTOCOL
1. `scripts/m3_advancement_baseline.py` (Phase 0: Baseline Reproduction)
2. `scripts/temporal_alert_policy.py` (Policy classes: Persistence, Hysteresis, Cooldown, Smoothing, EMA, Accumulation, Combined TAP)
3. `scripts/run_temporal_policy_ablation.py` (Phase 1: Validation Policy Sweep)
4. `scripts/select_validation_policy.py` (Validation Policy Selection $\arg\max U_{\text{val}}$)
5. `scripts/freeze_m3_temporal_policy.py` (Freeze Policy JSON)
6. `scripts/evaluate_frozen_policy_test.py` (Single-Pass Held-Out Test Evaluation)
7. `scripts/audit_m3_advancement.py` (Master Advancement Audit & Audit Verification)
