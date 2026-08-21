# 🔬 PHASE 1: DOMAIN SHIFT, CALIBRATION & HARD-CASE FORENSICS REPORT

**Development Source:** PhysioNet Set A (Emory University Hospital)  
**External Target Source:** PhysioNet Set B (BIDMC)  

---

## 1. Feature Distribution Shift (Canonical 8 Features)

```text
      Feature  Emory_Source_Mean  BIDMC_Target_Mean       SMD  KS_Statistic
          p_t           0.052347           0.053951  0.010754      0.138188
        ma_2h           0.049812           0.051756  0.013302      0.139489
        ma_6h           0.040366           0.043895  0.026472      0.141924
     slope_1h           0.005069           0.004391 -0.032712      0.077341
     accel_1h           0.000104          -0.000008 -0.004639      0.069751
 persist_th20           0.467132           0.668805  0.070286      0.011873
 occupancy_6h           0.059173           0.057616 -0.007071      0.005371
volatility_6h           0.008966           0.008442 -0.021587      0.140814
```

---

## 2. Hard-Case Composition Shift

```text
  Hard_Case_Subgroup  Emory_Val_Pct  BIDMC_Test_Pct
     Easy_Septic_Pct      66.863905       59.474672
Late_Weak_Septic_Pct      14.792899       16.510319
Invisible_Septic_Pct      18.343195       24.015009
 High_Risk_Mimic_Pct      27.774799       20.809126
```

---

## 3. Core Domain Shift Finding

> **Key Finding:**  
> Non-septic high-risk mimics account for **20.81%** of non-septic stays at BIDMC (3,940 patients), causing persistent false-alarm accumulation. Late/Weak + Invisible sepsis cases account for **40.53%** of BIDMC sepsis stays, incurring heavy missed-sepsis penalties (-2.00 pts/patient).
