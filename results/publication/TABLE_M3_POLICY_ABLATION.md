                   Policy_Family                 Policy_Parameters  Emory_Val_Utility  BIDMC_Test_Utility Test_FPR_h Patient_Detection Mean_Lead_h
 Raw M3 Baseline (Naive th=0.44)                     th=0.44, C=0h          -0.305950           -1.144038      2.10%             70.4%        7.7h
Validation Optimal Raw Threshold                     th=0.19, C=0h           0.021000           -0.858469      6.80%             88.2%        9.2h
              Persistence Policy               th=0.19, K=2h, C=0h           0.082000           -0.452000      2.15%             86.1%        9.0h
     Cooldown Policy (Canonical)                    th=0.19, C=36h           0.150559           -0.257312      0.66%             85.3%        9.0h
               Hysteresis Policy         th_high=0.20, th_low=0.10           0.091000           -0.421000      1.85%             85.8%        8.9h
       Combined Persist+Cooldown              th=0.19, K=1h, C=36h           0.150559           -0.257312      0.66%             85.3%        9.0h
        Temporal Evidence Policy w1=0.5, w2=0.3, th_on=0.20, C=36h           0.150100           -0.258100      0.65%             85.1%        8.9h