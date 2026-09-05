# 🛡️ NEXT-PHASE HOSTILE REVIEWER AUDIT (`NEXT_PHASE_REVIEWER_AUDIT.md`)

**Target Manuscript:** *A Time-Aware Transformer for Cross-Hospital Sepsis Early Warning: Linking Discrimination, Decision Utility, and Alert Burden*  
**Auditor Role:** Senior Biomedical Machine Learning Researcher, Statistical Auditor, & Clinical ML Journal Reviewer  
**Audit Date:** August 23, 2026  
**Repository State:** Frozen Checkpoint `best_m3_frozen.pt` (SHA256: `5b226074...`), Predictions `m3_final_test_predictions.npz` (SHA256: `e4a6a5e1...`)

---

## 1. Executive Reviewer Summary

This audit evaluates the scientific defensibility, methodological rigor, and claim discipline of the $M3$ Sepsis Early Warning manuscript from the perspective of an aggressive, top-tier clinical machine learning reviewer (e.g., *Lancet Digital Health*, *Nature Medicine*, *IEEE TBME*, *Journal of Biomedical Informatics*).

While the experimental core is strong—featuring a frozen model achieving an AUROC of **`0.961726`** and an official normalized PhysioNet utility of **`+0.655944`** on an independent $N=20,000$ external test cohort (Emory University Hospital)—the manuscript contains specific presentation and claim risks that must be addressed to ensure complete defense against hostile peer review.

---

## 2. Reviewer Attack Test Matrix (18 Core Concerns)

| Concern ID | Reviewer Concern | Empirical Repository Evidence | Current Manuscript Handling | Risk Level | Required Fix / Wording Adjustment |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **REV-01** | **Positive Utility vs. Low PPV** | $M3$ achieves $U_{\text{official}} = +0.655944$ despite PPV = $18.81\%$ ($4,333$ false alerts out of $5,337$). | Currently reports $U_{\text{official}}$ prominently without fully connecting it to the false alert volume in the abstract. | **CRITICAL** | Explicitly state PPV ($18.81\%$) and false alert count ($4,333$) alongside $U_{\text{official}}$ in the Abstract and Results. |
| **REV-02** | **High AUROC vs. Operational Alarm Fatigue** | AUROC = $0.9617$, but alert frequency is $16.99$ alerts / 100 patient-days ($25.86\%$ of patients alerted). | Mentions alert count in workload table, but discussion does not emphasize alarm fatigue. | **HIGH** | Add dedicated Discussion subsection on "Alarm Fatigue and Clinical Workload Implications". |
| **REV-03** | **Threshold Optimization & Test Leakage** | Threshold $th=0.190$ yields peak test utility ($+0.655944$). Reviewers will suspect test-set threshold tuning. | Documented in provenance file, but manuscript text might read as if $0.190$ was optimized on test data. | **CRITICAL** | Enforce exact phrasing: *"Among the evaluated thresholds in the sensitivity analysis, the prespecified validation threshold of 0.190 achieved the highest observed official utility on the independent Emory test cohort."* |
| **REV-04** | **Cross-Hospital Dataset Shift** | Transfer from BIDMC ($N=20,336$) to Emory ($N=20,000$). | Fully documented; Set A = BIDMC, Set B = Emory. | **LOW** | Preserve exact cohort provenance descriptions everywhere. |
| **REV-05** | **Baseline Utility Fairness** | XGBoost, Plain Transformer, GRU-D, TCN lack saved raw prediction arrays for official utility calculation. | Baseline table displays `—` for un-evaluated utilities. | **HIGH** | Retain `—` in official utility column for non-evaluated baselines and state explicitly that raw predictions were not preserved. |
| **REV-06** | **Utility Normalization Interpretation** | $U_{\text{official}} = \frac{U_{\text{obs}} - U_{\text{inact}}}{U_{\text{best}} - U_{\text{inact}}} = \frac{1514.78 - (-9512.44)}{7298.78 - (-9512.44)} = 0.655944$. | Mathematical formula included in Methods. | **MODERATE** | Clarify that $U_{\text{official}} = 0.655944$ means achieving $65.59\%$ of maximum possible utility above inaction, NOT clinical effectiveness. |
| **REV-07** | **Prevalence Shift Impact** | BIDMC sepsis prevalence = $8.80\%$; Emory sepsis prevalence = $5.33\%$. | Noted in Table 1 cohort provenance. | **MODERATE** | Discuss prevalence shift effect on PPV (lower test prevalence naturally depresses PPV). |
| **REV-08** | **Probability Calibration** | Brier score = $0.015290$, ECE = $0.018151$ (10 equal-width bins). | Reported in Table 2 benchmark. | **LOW** | Mention reliability diagrams and low ECE in Calibration subsection. |
| **REV-09** | **Missingness Mask & Time Delta Ablation** | $2 \times 2$ factorial ablation shows main effects: $m = +0.0155$, $\Delta t = +0.0215$ AUROC. | Reported in Table 3. | **LOW** | Highlight temporal sampling dynamics as the primary driver of representation strength. |
| **REV-10** | **Temporal Lead Time & Offsets** | Official evaluator applies $t_{\text{sepsis}} = t_{\text{label}} + 6\text{h}$ offset. | Included in Methods Section 3.4. | **MODERATE** | Explicitly define $t_{\text{sepsis}}$ shift logic to prevent confusion over prediction lead time. |
| **REV-11** | **Retrospective vs. Prospective Nature** | Retrospective analysis on PhysioNet 2019 challenge dataset ($40,336$ total stays). | Acknowledged in Methods. | **HIGH** | Add explicit Disclaimer: *"Positive utility does not equal proven prospective clinical efficacy."* |
| **REV-12** | **Limited Health System Scope (Two Centers)** | BIDMC (Set A) and Emory (Set B) only. | Stated in Cohort setup. | **MODERATE** | Note in Limitations that multi-center evaluation across $>2$ health systems remains future work. |
| **REV-13** | **Dataset-Specific Observation Workflows** | Hourly sampling rates differ between electronic health record systems. | Captured via $\Delta t$ feature. | **LOW** | Discuss how $\Delta t$ adapts embeddings to site-specific nursing protocols. |
| **REV-14** | **Overclaiming Clinical Readiness** | Language like "deployment-ready" or "clinically effective". | Audited in claim audit. | **CRITICAL** | Eliminate all occurrences of prohibited terms ("clinically ready", "proves", "state-of-the-art"). |
| **REV-15** | **Statistical Resampling Unit** | Patient-level bootstrap ($B=1,000$) vs hourly-level bootstrap. | Patient-level bootstrap verified. | **LOW** | State explicitly: *"Resampling was conducted at the patient stay level to preserve within-patient temporal dependencies."* |
| **REV-16** | **Initialization Stability Across Seeds** | 6-seed stability test ($AUROC = 0.9609 \pm 0.0016$, $Utility = +0.6559 \pm 0.0020$). | Reported in Table 4. | **LOW** | Highlight low standard deviation across random seeds. |
| **REV-17** | **Reproducibility Manifest & Hashes** | SHA256 hashes provided for checkpoint, prediction arrays, and split manifests. | Included in Reproducibility Manifest. | **LOW** | Provide cryptographic hashes in Supplementary Material. |
| **REV-18** | **Oracle Ceiling Interpretation** | Ground-truth oracle utility = $1.000000$ ($7,298.78$ pts). | Included in Table 5 decomposition. | **MODERATE** | Label Oracle Ceiling explicitly as an *infeasible label-informed mathematical reference bound*. |

---

## 3. Key Reviewer Objections & Defensible Rebuttals

### Objection 1: "Your model has a PPV of only 18.81%—four out of five alerts are false alarms. How can you claim high clinical utility?"
- **Defensible Rebuttal:**  
  *"We do not claim that an 18.81% PPV is optimal for all clinical environments. Rather, our evaluation under the official PhysioNet 2019 scoring framework demonstrates that early warning models operate under an intrinsic trade-off between early true positive detection (rewarded up to +1.0 points) and false alarm frequency (penalized at -0.05 points/hour). The positive utility score of +0.655944 reflects that the clinical benefit of early sepsis identification (up to 12 hours prior to clinical onset) outweighs the bounded penalty of false alerts under the challenge's predefined decision-cost function. We explicitly highlight this 18.81% PPV and 16.99 alerts/100 patient-days workload burden as a key operational constraint requiring prospective workflow design."*

### Objection 2: "Did you tune your threshold on the external test set to get a positive utility?"
- **Defensible Rebuttal:**  
  *"No. The threshold th=0.190 was selected strictly on the Set A (BIDMC) validation cohort (N=4,144) by maximizing validation utility, and was locked prior to unblinding external test predictions. Applying this frozen threshold once to the independent Set B (Emory) test cohort (N=20,000) yielded U_official = +0.655944. Subsequent sensitivity analysis across thresholds th in [0.05, 0.70] confirmed that the validation-selected threshold of 0.190 achieved the highest observed official utility on the external test cohort."*

### Objection 3: "Why is the baseline utility column filled with dashes ('—') for XGBoost, Plain Transformer, and GRU-D?"
- **Defensible Rebuttal:**  
  *"In accordance with strict scientific reproducibility standards, we only report official PhysioNet utility values for models where full, un-truncated hourly prediction arrays were saved and verified. For baseline models where only summary discrimination metrics (AUROC/AUPRC) were preserved from historical phases, we display '—' rather than estimating or fabricating unverified utility figures."*

---

## 4. Final Peer-Review Audit Verdict

```text
================================================================================
                    PEER-REVIEW AUDIT VERDICT: DEFENSIVE
================================================================================
   DEFENSIBILITY SCORE : 94 / 100
   STATUS               : DEFENSIVE WITH QUALIFICATIONS
   RECOMMENDED ACTION   : Execute textual claim adjustments in MANUSCRIPT_REVISION_ACTION_PLAN.md
================================================================================
```
