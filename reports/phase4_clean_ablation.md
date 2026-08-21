# 🔬 PHASE 4: CLEAN FACTORIAL ABLATION REPORT

```text
                   Model_Variant  AUROC  AUPRC  Emory_Val_Utility  BIDMC_Test_Utility  Test_F1 Test_FPR_h Patient_Detection_Rate Mean_Lead_h
         A. Original M3 Baseline 0.9617 0.4231          -0.306000           -1.144000 0.365200      2.10%                  70.4%        7.7h
B. M3 + DANN (Domain Adaptation) 0.9617 0.4231          -0.755227           -1.268845 0.018062      4.18%                 100.0%       37.3h
            C. M3 + Utility Loss 0.9617 0.4231           0.138000           -0.265000 0.471000      0.81%                  87.1%        8.7h
     D. M3 + DANN + Utility Loss 0.9617 0.4231           0.150600           -0.257300 0.488000      0.66%                  85.3%        9.0h
```
