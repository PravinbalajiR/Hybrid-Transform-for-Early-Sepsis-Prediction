# Final Statistical Consistency & Leakage Audit

**Audit Date:** 2026-08-15  
**Audited Models:** M1, M2, M3, M4, M5  
**Data Isolation Verification:** **PASSED (Zero Patient Overlap)**  
**Threshold Locking Verification:** **PASSED (Validation-Only Threshold Selection)**  

---

## 1. Statistical Consistency Resolution

During the M5 audit, a reporting label inconsistency was identified:
- **Reported Metric Difference**: $\Delta 	ext{AUROC} = -0.0274$ (95% CI: `[-0.0490, -0.0095]`)
- **Previous Label**: `Statistically Significant: NO`

### **Mathematical Resolution**:
- Because the entire 95% Confidence Interval is **strictly negative (`< 0`)**, the difference is **statistically significant ($lpha = 0.05$)**.
- **Corrected Reporting Label**: `Statistically Significant Difference: YES (M3 is statistically superior to M5)`.
- **Note**: Zero underlying model weights, probabilities, or metrics were modified. Only the natural language reporting string was corrected.

---

## 2. Data Leakage & Reproducibility Matrix

| Model | Patient Split Isolation | Normalizer Fit | Threshold Selection | Checkpoint SHA256 | Reproducibility Status |
|---|:---:|:---:|:---:|:---:|:---:|
| **M1 (XGBoost)** | PASS | Train Only | Validation Only | `N/A (Script)` | FULLY REPRODUCIBLE |
| **M2 (Plain Transformer)** | PASS | Train Only | Validation Only | `88a1b...` | FULLY REPRODUCIBLE |
| **M3 (Time-Aware Trans.)** | PASS | Train Only | Validation Only | `5b22607444f4a242a52d...` | **FULLY REPRODUCIBLE (PRIMARY)** |
| **M4 (Organ Hybrid)** | PASS | Train Only | Validation Only | `4c91a...` | HISTORICAL / ABLATION |
| **M5 (Multi-Hybrid)** | PASS | Train Only | Validation Only | `e3b9f...` | **FULLY REPRODUCIBLE (ABLATION)** |
