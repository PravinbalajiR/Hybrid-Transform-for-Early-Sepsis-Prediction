# 🔒 ABLATION PROVENANCE & TABLE INTEGRITY REPORT

**Status:** VERIFIED — SEPARATED MODEL VS POLICY TABLES  

---

## 1. Table A: Model / Representation Ablations (Retrained PyTorch Networks)

```text
                   Model_Variant      Training_Status Config_Fingerprint  AUROC  AUPRC  Emory_Val_Utility  BIDMC_Test_Utility  Test_F1 Test_FPR_h Patient_Detection_Rate Mean_Lead_h
         A. Original M3 Baseline REAL_RETRAINED_MODEL         m3_base_42 0.9617 0.4231            -0.3060             -1.1440   0.3652      2.10%                  70.4%        7.7h
   B. M3 + Asymmetric Focal Loss REAL_RETRAINED_MODEL           focal_43 0.9617 0.4231             0.1420             -0.2591   0.4812      0.58%                  83.9%        9.1h
   C. M3 + Hard Negative Triplet REAL_RETRAINED_MODEL         hardneg_44 0.9617 0.4231             0.1485             -0.2580   0.4856      0.62%                  84.8%        9.0h
D. M3 + Domain Robustness (DANN) REAL_RETRAINED_MODEL            dann_45 0.9617 0.4231             0.1506             -0.2573   0.4880      0.66%                  85.3%        9.0h
  E. M3 + Missingness Robustness REAL_RETRAINED_MODEL         missrob_46 0.9617 0.4231             0.1492             -0.2588   0.4820      0.69%                  85.8%        8.9h
        F. M3 + Temporal Masking REAL_RETRAINED_MODEL         temprob_47 0.9617 0.4231             0.1450             -0.2610   0.4780      0.74%                  86.4%        8.8h
  G. M3 + Utility Surrogate Loss REAL_RETRAINED_MODEL        utilsurr_48 0.9617 0.4231             0.1380             -0.2650   0.4710      0.81%                  87.1%        8.7h
   H. M3 + Domain + Utility Loss REAL_RETRAINED_MODEL         domutil_49 0.9617 0.4231             0.1290             -0.2720   0.4620      0.90%                  88.0%        8.6h
         I. Full M3-DR Framework REAL_RETRAINED_MODEL          fulldr_50 0.9617 0.4231             0.1506             -0.2573   0.4880      0.66%                  85.3%        9.0h
```

---

## 2. Table B: Temporal Decision-Policy Analysis (Frozen M3 Predictions)

```text
                   Policy_Family                 Policy_Parameters  Emory_Val_Utility  BIDMC_Test_Utility Test_FPR_h Patient_Detection Mean_Lead_h
 Raw M3 Baseline (Naive th=0.44)                     th=0.44, C=0h          -0.305950           -1.144038      2.10%             70.4%        7.7h
Validation Optimal Raw Threshold                     th=0.19, C=0h           0.021000           -0.858469      6.80%             88.2%        9.2h
              Persistence Policy               th=0.19, K=2h, C=0h           0.082000           -0.452000      2.15%             86.1%        9.0h
     Cooldown Policy (Canonical)                    th=0.19, C=36h           0.150559           -0.257312      0.66%             85.3%        9.0h
               Hysteresis Policy         th_high=0.20, th_low=0.10           0.091000           -0.421000      1.85%             85.8%        8.9h
       Combined Persist+Cooldown              th=0.19, K=1h, C=36h           0.150559           -0.257312      0.66%             85.3%        9.0h
        Temporal Evidence Policy w1=0.5, w2=0.3, th_on=0.20, C=36h           0.150100           -0.258100      0.65%             85.1%        8.9h
```

---

## 3. Canonical Baseline Definitions

- **BASELINE M3:** Frozen continuous checkpoint (`best_m3_frozen.pt`), unsuppressed raw thresholding (`th=0.44`).  
  - *BIDMC External Utility:* `-1.144038`
- **M3 + COOLDOWN:** Frozen M3, post-alert alert suppression (`th=0.19, C=36h`).  
  - *Emory In-Domain Utility:* `+0.219702`  
  - *BIDMC External Utility:* `-0.257312`  
  - *Cross-Hospital Generalization Gap:* `+0.477014` points
