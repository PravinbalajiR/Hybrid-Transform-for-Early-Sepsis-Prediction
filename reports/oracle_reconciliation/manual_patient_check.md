# 🧮 STEP-BY-STEP MANUAL PATIENT UTILITY ARITHMETIC TRACES

This document contains fully worked, step-by-step arithmetic traces for 5 representative BIDMC patients, demonstrating 100% exact numerical agreement ($\le 10^{-10}$ discrepancy) between the official repository scorer and the independent zero-dependency calculator.

---

## 1. Patient Check 1: Patient ID 46 (Septic, Perfect 6-Hour Early Alarm)

### A. Patient Profile
- **Cohort:** BIDMC Test Set
- **Length ($T$):** 36 hours
- **Is Septic:** YES ($y_t = 1$ starting at $t=26$)
- **True Sepsis Onset ($t_{\text{onset}}$):** Hour 26
- **Model Action ($P$):** First alarm issued at $t_{\text{alarm}} = 20$
- **Calculated Lead Time ($\Delta t$):** $t_{\text{onset}} - t_{\text{alarm}} = 26 - 20 = 6.0$ hours

### B. Step-by-Step Manual Arithmetic Trace
1. **Determine Sepsis State:** Patient is septic $\implies$ Best possible utility = $+1.0$.
2. **Evaluate First Alarm Timing:** $\Delta t = 6.0$ hours.
3. **Apply Timing Reward Rule:** Since $\Delta t = dt_{\text{optimal}} = 6.0$, full true-positive credit is awarded:
   $$\text{TP Reward} = +1.000000$$
4. **Evaluate Early False Alarms:** Early alarm cutoff is $t_{\text{onset}} - 12 = 26 - 12 = 14$. Alarm times are at $t=20 \ge 14 \implies N_{\text{early\_fp}} = 0$.
   $$\text{FP Penalty} = 0 \times (-0.05) = 0.000000$$
5. **Sum Total Achieved Utility:**
   $$\text{Achieved Utility} = \text{TP Reward} + \text{FP Penalty} = +1.000000 + 0.000000 = +1.000000$$

### C. System Verification Comparison
- **Independent Calculator:** `+1.000000`
- **Official Scorer:** `+1.000000`
- **Discrepancy:** `0.000000e+00` (**EXACT MATCH**)

---

## 2. Patient Check 2: Patient ID 12 (Septic, 8-Hour Early Alarm with Linear Decay)

### A. Patient Profile
- **Cohort:** BIDMC Test Set
- **Length ($T$):** 90 hours
- **Is Septic:** YES ($y_t = 1$ starting at $t=80$)
- **True Sepsis Onset ($t_{\text{onset}}$):** Hour 80
- **Model Action ($P$):** First alarm issued at $t_{\text{alarm}} = 72$
- **Calculated Lead Time ($\Delta t$):** $t_{\text{onset}} - t_{\text{alarm}} = 80 - 72 = 8.0$ hours

### B. Step-by-Step Manual Arithmetic Trace
1. **Determine Sepsis State:** Patient is septic $\implies$ Best possible utility = $+1.0$.
2. **Evaluate First Alarm Timing:** $\Delta t = 8.0$ hours ($6.0 \le \Delta t < 12.0$).
3. **Apply Timing Reward Rule:** Linear decay formula for early warning window:
   $$\text{TP Reward} = 1.0 \times \frac{12.0 - \Delta t}{12.0 - 6.0} = \frac{12.0 - 8.0}{6.0} = \frac{4.0}{6.0} = +0.666666666667$$
4. **Evaluate Early False Alarms:** Early alarm cutoff is $80 - 12 = 68$. Alarm is at $t=72 > 68 \implies N_{\text{early\_fp}} = 0$.
   $$\text{FP Penalty} = 0.000000$$
5. **Sum Total Achieved Utility:**
   $$\text{Achieved Utility} = +0.666666666667 + 0.000000 = +0.666666666667$$

### C. System Verification Comparison
- **Independent Calculator:** `+0.666667`
- **Official Scorer:** `+0.666667`
- **Discrepancy:** `0.000000e+00` (**EXACT MATCH**)

---

## 3. Patient Check 3: Patient ID 54 (Septic, Immediate Onset at $t=0$)

### A. Patient Profile
- **Cohort:** BIDMC Test Set
- **Length ($T$):** 8 hours
- **Is Septic:** YES ($y_t = 1$ starting at $t=0$)
- **True Sepsis Onset ($t_{\text{onset}}$):** Hour 0
- **Model Action ($P$):** First alarm issued at $t_{\text{alarm}} = 0$
- **Calculated Lead Time ($\Delta t$):** $0 - 0 = 0.0$ hours

### B. Step-by-Step Manual Arithmetic Trace
1. **Determine Sepsis State:** Patient is septic $\implies$ Best possible utility = $+1.0$.
2. **Evaluate First Alarm Timing:** $\Delta t = 0.0$ hours (onset alarm, $-3.0 \le \Delta t < 6.0$).
3. **Apply Timing Reward Rule:** Decay formula for optimal/late window:
   $$\text{TP Reward} = 1.0 \times \frac{\Delta t + 3.0}{6.0 + 3.0} = \frac{0.0 + 3.0}{9.0} = \frac{3.0}{9.0} = +0.333333333333$$
4. **Evaluate Early False Alarms:** Cutoff $0 - 12 = -12 \implies N_{\text{early\_fp}} = 0$.
5. **Sum Total Achieved Utility:**
   $$\text{Achieved Utility} = +0.333333333333$$

### C. System Verification Comparison
- **Independent Calculator:** `+0.333333`
- **Official Scorer:** `+0.333333`
- **Discrepancy:** `0.000000e+00` (**EXACT MATCH**)

---

## 4. Patient Check 4: Patient ID 355 (Septic, Missed Sepsis / No Alarm)

### A. Patient Profile
- **Cohort:** BIDMC Test Set
- **Length ($T$):** 16 hours
- **Is Septic:** YES ($y_t = 1$ starting at $t=6$)
- **True Sepsis Onset ($t_{\text{onset}}$):** Hour 6
- **Model Action ($P$):** No alarms issued ($P = \mathbf{0}$)

### B. Step-by-Step Manual Arithmetic Trace
1. **Determine Sepsis State:** Patient is septic $\implies$ Best possible utility = $+1.0$.
2. **Evaluate First Alarm Timing:** $t_{\text{alarm}} = \text{None}$.
3. **Apply Missed Sepsis Rule:** Missed sepsis penalty is directly assigned:
   $$\text{Achieved Utility} = u_{\text{fn}} = -2.000000$$

### C. System Verification Comparison
- **Independent Calculator:** `-2.000000`
- **Official Scorer:** `-2.000000`
- **Discrepancy:** `0.000000e+00` (**EXACT MATCH**)

---

## 5. Patient Check 5: Patient ID 15 (Non-Septic Mimic, 3 False Alarm Hours)

### A. Patient Profile
- **Cohort:** BIDMC Test Set
- **Length ($T$):** 168 hours
- **Is Septic:** NO ($y_t = 0 \forall t$)
- **True Sepsis Onset ($t_{\text{onset}}$):** None
- **Model Action ($P$):** Alarms issued for 3 hours ($N_{\text{fp}} = 3$)

### B. Step-by-Step Manual Arithmetic Trace
1. **Determine Sepsis State:** Non-septic patient $\implies$ Best possible utility = $0.0$.
2. **Calculate False Alarm Hours:** $N_{\text{fp}} = 3$.
3. **Apply False Alarm Penalty Rule:** Each FP hour penalized by $-0.05$:
   $$\text{Achieved Utility} = 3 \times (-0.05) = -0.150000$$

### C. System Verification Comparison
- **Independent Calculator:** `-0.150000`
- **Official Scorer:** `-0.150000`
- **Discrepancy:** `0.000000e+00` (**EXACT MATCH**)

---

## 📋 Full 10-Patient Verification Table

| Patient ID | Is Septic | Length (h) | Onset (h) | First Alarm (h) | Lead Time (h) | Official Utility | Independent Utility | Discrepancy | TP Reward | FN Penalty | FP Penalty |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **12** | 1 | 90 | 80 | 72 | 8 | +0.666667 | +0.666667 | 0.00e+00 | +0.666667 | 0.00 | 0.00 |
| **54** | 1 | 8 | 0 | 0 | 0 | +0.333333 | +0.333333 | 0.00e+00 | +0.333333 | 0.00 | 0.00 |
| **355** | 1 | 16 | 6 | -1 | -999 | -2.000000 | -2.000000 | 0.00e+00 | 0.000000 | -2.00 | 0.00 |
| **46** | 1 | 36 | 26 | 20 | 6 | +1.000000 | +1.000000 | 0.00e+00 | +1.000000 | 0.00 | 0.00 |
| **15** | 0 | 168 | -1 | 11 | -999 | -0.150000 | -0.150000 | 0.00e+00 | 0.000000 | 0.00 | -0.15 |
| **14** | 0 | 16 | -1 | 0 | -999 | -0.050000 | -0.050000 | 0.00e+00 | 0.000000 | 0.00 | -0.05 |
| **39** | 0 | 21 | -1 | 19 | -999 | -0.050000 | -0.050000 | 0.00e+00 | 0.000000 | 0.00 | -0.05 |
| **3** | 0 | 59 | -1 | -1 | -999 | 0.000000 | 0.000000 | 0.00e+00 | 0.000000 | 0.00 | 0.00 |
| **11** | 0 | 43 | -1 | -1 | -999 | 0.000000 | 0.000000 | 0.00e+00 | 0.000000 | 0.00 | 0.00 |
| **16** | 0 | 21 | -1 | -1 | -999 | 0.000000 | 0.000000 | 0.00000 | 0.000000 | 0.00 | 0.00 |
