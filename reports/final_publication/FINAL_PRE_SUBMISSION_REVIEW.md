# 🔬 PRE-SUBMISSION EXPERIMENTAL AUDIT & PROVENANCE REPORT

**Current Status:** **SCIENTIFIC FREEZE OPEN — EXPERIMENTAL VALIDATION & THRESHOLD AUDIT IN PROGRESS**  
**Target Manuscript:** [`paper/manuscript/FULL_MANUSCRIPT_DRAFT.md`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/paper/manuscript/FULL_MANUSCRIPT_DRAFT.md)  
**Date:** 2026-08-15  

---

## ⚠️ EXECUTIVE STATUS & AUDIT GATES

The manuscript scientific freeze has been **REOPENED**. Journal submission packaging is paused pending resolution of two critical experimental audit gates:

| Audit Gate | Status | Diagnostic Summary & Required Actions |
|---|:---:---|---|
| **Gate A: Operating-Point & Utility Validity** | 🟡 **IN PROGRESS** | The existing experiment logs applied a fixed global threshold ($th=0.60$) across all models, producing negative utility scores (M1=-1.42, M2=-1.15, M3=-0.95, M4=-1.84, M5=-2.56) that are worse than doing nothing ($0.0$). **Action:** Execute per-model validation threshold optimization ($th_{\text{val\_opt}}$) on real saved model predictions before locking operational metrics. |
| **Gate B: Prediction Provenance & AUROC Integrity** | 🟡 **IN PROGRESS** | Verify exact checkpoint provenance (`experiments/final_m3_frozen/best_m3_frozen.pt`, SHA256: `5b226074...`) and patient-level splitting ($N_{\text{test}}=20,000$) to guarantee zero temporal or feature leakage for M3's reported 0.9617 AUROC and 0.4231 AUPRC. |

---

## 🛠️ GATE A: Operating-Point Thresholding Protocol

To eliminate the fixed-threshold artifact, the evaluation pipeline enforces strict per-model threshold selection:

```text
1. Validation Cohort Predictions (N = 2,034):
   For each model M_i ∈ {M1, M2, M3-Delta, M3-Mask, M3-Full, M4, M5}:
     Grid-search th ∈ [0.01, 0.99] to maximize Validation PhysioNet Utility (U_val).
     Lock model-specific threshold th_val_opt[M_i].

2. Held-Out Test Cohort Predictions (N = 20,000):
   Evaluate th_val_opt[M_i] single-pass on test predictions.
   Record test Utility, F1, Precision, Recall, FPR/h, and Lead Time.
```

*Note: Synthetic diagnostic tests confirmed that per-model thresholding resolves negative utility artifacts when probability calibrations differ across architectures. Real prediction arrays must be evaluated through this pipeline before canonical re-freezing.*

---

## 🛡️ GATE B: Checkpoint & Data Leakage Audit

- **Checkpoint Audit:**  
  - Primary model checkpoint: `experiments/final_m3_frozen/best_m3_frozen.pt` (SHA256: `5b22607444f4a242a52d0d9337e60c4c63044542dc6796a4a9de78c5ef38057c`).
- **Splitting Isolation:**  
  - Patient-level split: 18,302 Train / 2,034 Validation / 20,000 Test (zero patient overlap).
  - Preprocessing Z-scores: fit strictly on Train partition.

---

## 🏁 NEXT STEPS BEFORE RE-FREEZING

1. Execute per-model validation threshold search on real prediction files.
2. Verify positive test utility scores and update canonical JSON (`CANONICAL_NUMERICAL_RESULTS.json`).
3. Re-align manuscript sections 1–5, tables 1–3, and abstract with real threshold-optimized metrics.
4. Obtain final verification pass before journal packaging.
