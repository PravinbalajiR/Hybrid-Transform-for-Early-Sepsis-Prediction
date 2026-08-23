# 📍 THRESHOLD SELECTION PROVENANCE & OPERATING POLICY AUDIT

This document provides a complete audit of the operating threshold selection procedure, provenance, and policy isolation for the primary $M3$ Time-Aware Transformer model.

---

## 1. Operating Threshold Selection Procedure

1. **Cohort Used for Selection:**
   - Selection Cohort: Set A (Beth Israel Deaconess Medical Center / BIDMC) validation split ($N = 4,144$ ICU stays).
   - Test Cohort Access: **ZERO** test set data from Set B (Emory University Hospital) was accessed or used during threshold tuning.
2. **Selection Criterion:**
   - Maximized empirical utility score on BIDMC validation predictions across candidate thresholds $th \in [0.01, 0.99]$ in steps of $0.005$.
3. **Selected Prespecified Threshold:**
   - **Validation-Selected Threshold ($th^*$):** **`0.190`**
4. **Freezing & External Deployment:**
   - The threshold $th^* = 0.190$ was frozen prior to unblinding external test predictions.
   - The frozen model and locked threshold $th = 0.190$ were applied **ONCE** to the held-out Emory test cohort ($N = 20,000$).

---

## 2. External Test Evaluation vs. Sensitivity Analysis

- **Legitimate External Test Evaluation:**
  - Prespecified validation threshold $th = 0.190$ evaluated on Emory external test set ($N=20,000$).
  - **Official Normalized PhysioNet Utility:** **`0.655944`** (`+0.6559`).
  - **Test AUROC:** **`0.961726`** (`0.9617`).
  - **Test AUPRC:** **`0.423114`** (`0.4231`).
  - **Accuracy:** `0.971542` (`97.15%`).
  - **F-measure:** `0.231804`.
- **External Threshold Sensitivity Analysis:**
  - Post-hoc sensitivity sweep across $th \in [0.05, 0.70]$ performed strictly to evaluate utility sensitivity to decision boundary shifts.
  - **Observed Peak:** Among the evaluated thresholds in the sensitivity analysis, the prespecified validation threshold of $0.190$ achieved the highest observed official utility ($0.655944$) on the independent Emory test cohort.
