# 🔬 PHASE 0 EXPERIMENT FORENSICS & INVENTORY REPORT

**Report Date:** 2026-08-21 21:14:10  
**Base Checkpoint SHA256:** `5b22607444f4a242a52d0d9337e60c4c63044542dc6796a4a9de78c5ef38057c`  
**Test Prediction SHA256:** `02fd6eb78682be8ca5743c4b3fddfcc7f57ed56f27f8496092108c30b2188a3d`  

---

## 1. Classification Summary

- **Total Claimed Experiments Audited:** 13
- **Real Retrained Models (Base M3 + Isolated Phase 12.5 Suite):** 2
- **Policy / Post-Processing Sweeps (Phases 2-12 Wrapper Layers):** 11

---

## 2. Full Experiment Inventory & Scientific Classification Table

```text
     Phase              Experiment_Name       Classification                     Actual_Training_Performed                             Script_File
   Phase 1              Raw M3 Baseline REAL_RETRAINED_MODEL               YES (Canonical Base Checkpoint)              m3_advancement_baseline.py
   Phase 2         M3 + Cooldown Policy          POLICY_ONLY                                            NO      run_m3_tap_phase3_policy_search.py
   Phase 3       M3-TAP Pareto Frontier          POLICY_ONLY                                            NO         analyze_m3_tap_phase3_pareto.py
   Phase 4            M3 + U-TRC Policy          POLICY_ONLY                                            NO          run_m3_phase4_temporal_risk.py
   Phase 5          M3 + HTR Specialist          POLICY_ONLY NO (Evaluated as Policy Layer over Frozen M3)                    run_m3_phase5_htr.py
   Phase 6 Utility Feasibility Analysis          POLICY_ONLY                                            NO            run_m3_phase6_feasibility.py
   Phase 7    M3 + U-TRL Representation          POLICY_ONLY NO (Evaluated as Policy Layer over Frozen M3)                   run_m3_phase7_utrl.py
   Phase 8      M3-UAT Multi-Task Model          POLICY_ONLY NO (Evaluated as Policy Layer over Frozen M3)                    run_m3_phase8_uat.py
   Phase 9        M3 UBPG Policy Sweeps          POLICY_ONLY                                            NO                   run_m3_phase9_ubpg.py
  Phase 10            Shift Diagnostics          POLICY_ONLY                                            NO           run_m3_phase10_diagnostics.py
  Phase 11     M3-SR Shift-Robust Model          POLICY_ONLY NO (Evaluated as Policy Layer over Frozen M3)                  run_m3_phase11_m3sr.py
  Phase 12    M3-DR Domain-Robust Model          POLICY_ONLY NO (Evaluated as Policy Layer over Frozen M3) run_m3_phase12_domain_generalization.py
Phase 12.5   M3 Phase 12.5 Forensic Fix REAL_RETRAINED_MODEL            YES (Isolated A-I Retrained Suite)            run_m3_phase12_5_forensic.py
```

---

## 3. Scientific Verification & Publication Table Separation Decision

1. **Table A (Model Ablation):** Will contain ONLY genuinely retrained PyTorch models with independent checkpoints and non-identical parameter distances.
2. **Table B (Policy Ablation):** Will contain all temporal decision policy sweeps (`Raw`, `Cooldown`, `Persistence`, `Hysteresis`, `Combined`, `Evidence`).
