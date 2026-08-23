# 🏆 FINAL SUBMISSION VALIDATION REPORT (`FINAL_SUBMISSION_VALIDATION_REPORT.md`)

**Repository:** `Hybrid-Transform-for-Early-Sepsis-Prediction`  
**Branch:** `paper-v1.0`  
**Git Commit SHA:** `46dd580`  
**Evaluation Engine:** Official PhysioNet 2019 Evaluator (`evaluation/official_physionet2019.py`)  
**Auditor:** Senior Biomedical Machine Learning Researcher & Scientific Submission Auditor  
**Audit Date:** August 23, 2026

---

## A. Manuscript Files Revised

The following four primary manuscript files were updated to enforce 100% claim discipline, threshold isolation, operational workload disclosure, and baseline fairness:
1. [`paper/manuscript/final_publication_manuscript.md`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/paper/manuscript/final_publication_manuscript.md)
2. [`paper/manuscript/final_publication_manuscript.tex`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/paper/manuscript/final_publication_manuscript.tex)
3. [`paper/manuscript/full_research_paper.md`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/paper/manuscript/full_research_paper.md)
4. [`paper/manuscript/full_research_paper.tex`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/paper/manuscript/full_research_paper.tex)

---

## B. Numerical Consistency Audit

Every authoritative numerical metric in the frozen $M3$ research freeze has been audited and confirmed 100% consistent across all manuscript files:

| Metric Item | Authoritative Value | 95% Bootstrap CI | Verified Across All Files? | Source Artifact |
| :--- | :---: | :---: | :---: | :--- |
| **Test AUROC** | **`0.961726`** (`0.9617`) | `[0.9582, 0.9651]` | **VERIFIED** | [`results/m3_final_test_predictions.npz`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/results/m3_final_test_predictions.npz) |
| **Test AUPRC** | **`0.423114`** (`0.4231`) | `[0.4015, 0.4448]` | **VERIFIED** | [`results/m3_final_test_predictions.npz`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/results/m3_final_test_predictions.npz) |
| **Brier Score** | **`0.015290`** | `[0.0145, 0.0161]` | **VERIFIED** | [`results/revised_publication/FINAL_REPRODUCIBILITY_REPORT.md`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/results/revised_publication/FINAL_REPRODUCIBILITY_REPORT.md) |
| **ECE (10 Bins)** | **`0.018151`** | `[0.0162, 0.0201]` | **VERIFIED** | [`results/revised_publication/FINAL_REPRODUCIBILITY_REPORT.md`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/results/revised_publication/FINAL_REPRODUCIBILITY_REPORT.md) |
| **Accuracy ($th=0.190$)** | **`0.971542`** (`97.15%`) | `[0.9701, 0.9729]` | **VERIFIED** | `evaluate_sepsis_score.py` on Emory PSVs |
| **F-measure** | **`0.231804`** | `[0.2210, 0.2425]` | **VERIFIED** | `evaluate_sepsis_score.py` on Emory PSVs |
| **Raw Utility ($U_{\text{obs}}$)** | **`1515.6500`** pts | — | **VERIFIED** | `evaluate_sepsis_score.py` on Emory PSVs |
| **Inaction Utility ($U_{\text{inact}}$)** | **`-9512.4444`** pts | — | **VERIFIED** | `evaluate_sepsis_score.py` on Emory PSVs |
| **Oracle Utility ($U_{\text{best}}$)** | **`7298.7778`** pts | — | **VERIFIED** | `evaluate_sepsis_score.py` on Emory PSVs |
| **Official Utility ($U_{\text{official}}$)** | **`+0.655944`** (`+0.6559`) | `[+0.6310, +0.6800]` | **VERIFIED** | `evaluate_sepsis_score.py` ($th=0.190$) |
| **Development Cohort (Set A)** | **`20,336`** stays | — | **VERIFIED** | [`data/splits/train_ids.json`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/data/splits/train_ids.json) ($16,192$ train, $4,144$ val) |
| **External Test Cohort (Set B)** | **`20,000`** stays | — | **VERIFIED** | [`data/splits/test_ids.json`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/data/splits/test_ids.json) ($1,066$ septic, $18,934$ non-septic) |
| **Hourly Observations** | **`753,927`** hours | — | **VERIFIED** | Emory PSV data directory |
| **Total Alerts Issued** | **`5,337`** alerts | — | **VERIFIED** | $1,004$ True Positive, $4,333$ False Positive |
| **Alert PPV** | **`18.81%`** (`0.188121`) | — | **VERIFIED** | [`results/revised_publication/workload_operational_metrics.csv`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/results/revised_publication/workload_operational_metrics.csv) |
| **Alert Frequency** | **`16.99` alerts / 100 days** | — | **VERIFIED** | [`results/revised_publication/workload_operational_metrics.csv`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/results/revised_publication/workload_operational_metrics.csv) |
| **Patient Alert Rate** | **`25.86%`** ($5,172$/$20,000$) | — | **VERIFIED** | [`results/revised_publication/workload_operational_metrics.csv`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/results/revised_publication/workload_operational_metrics.csv) |
| **Prespecified Threshold** | **`0.190`** | — | **VERIFIED** | [`results/m3_selected_thresholds.json`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/results/m3_selected_thresholds.json) |

---

## C. Threshold Isolation Audit

The two-stage decision threshold isolation process has been verified:

$$\text{BIDMC Train } (N=16,192) \longrightarrow \text{BIDMC Val } (N=4,144) \longrightarrow \text{Grid Search } th \in [0.01, 0.99] \longrightarrow \text{Locked } th^*=0.190 \longrightarrow \text{Emory External Test } (N=20,000)$$

- **Stage 1 (Validation Selection):** Candidate thresholds $th \in [0.01, 0.99]$ were evaluated strictly on BIDMC validation predictions ($N=4,144$), identifying $th^*=0.190$ as the utility-maximizing threshold prior to unblinding external test predictions.
- **Stage 2 (External Test Evaluation):** The frozen model (`best_m3_frozen.pt`) and locked threshold $th=0.190$ were evaluated **once** on the independent held-out Emory test cohort ($N=20,000$).
- **Mandatory Phrasing Enforced:**
  > *"Among the evaluated thresholds in the sensitivity analysis, the prespecified validation threshold of 0.190 achieved the highest observed official utility on the independent Emory test cohort."*

---

## D. Negative & Unfavorable Findings Disclosure Audit

All 7 genuine unfavorable repository findings are explicitly disclosed across the revised manuscript suite:

1. **Low Precision (PPV = 18.81%):** Disclosed in Abstract, Results Section 4.5 (Table 4), and Discussion Section 5.5 ($4,333$ false alerts out of $5,337$ total alerts; $81.19\%$ false alarm rate).
2. **Substantial Operational Alert Burden:** Disclosed in Abstract, Results Section 4.5 ($16.99$ alerts / 100 patient-days; $25.86\%$ patients alerted), and Discussion Section 5.5.
3. **Architectural Saturation ($M4/M5$ Null Result):** Disclosed in Results Section 4.7 & Table 2/Table 6 (Organ-Aware $M4$ AUROC = $0.9582$, MoE $M5$ AUROC = $0.9591$ do not significantly outperform compact $M3$ $0.9617$, $p=0.068$).
4. **Adaptive Threshold Predictability Failure:** Disclosed in Results Section 4.8 & Discussion Section 5.7 (Classifier predicting custom threshold needs achieved AUPRC = $0.2653$ vs base rate $0.2608$, proving adaptive threshold needs are not predictable from early features).
5. **Sharp Utility Degradation Outside Threshold Range:** Disclosed in Results Section 4.4 (Table 2 sensitivity sweep showing $U=+0.5204$ at $th=0.05$ and $U=+0.5174$ at $th=0.70$).
6. **Cross-Hospital Prevalence Shift Impact:** Disclosed in Results Section 3.1 & Discussion Section 5.4 (BIDMC $8.80\%$ prevalence vs Emory $5.33\%$ prevalence depressing alert PPV).
7. **Unavailable Baseline Prediction Arrays:** Disclosed in Results Section 4.1 & Table 1 footnote (XGBoost, Plain Transformer, GRU-D, TCN baseline utilities displayed as `—` because raw prediction arrays were not preserved).

---

## E. Baseline Fairness Audit

- **Baseline Utility Column:** Enforced `—` for all baseline architectures without preserved `.npz` prediction arrays.
- **Footnote Enforced:** *"`—` indicates baseline models for which raw hourly prediction arrays were not preserved, preventing independent recomputation under the official PhysioNet 2019 utility evaluator (`evaluate_sepsis_score.py`) without re-training."*

---

## F. Scientific Claim Audit

All manuscript claims were audited and classified:

- **Verified Claims (Preserved):**
  - $M3$ Time-Aware Transformer achieves top cross-hospital discrimination (AUROC = $0.961726$, AUPRC = $0.423114$).
  - $M3$ achieves positive official normalized utility ($U_{\text{official}} = +0.655944$, 95% CI: `[+0.6310, +0.6800]`) under the official challenge metric at prespecified $th=0.190$.
  - Explicit missingness masks ($m$) and time deltas ($\Delta t$) drive discriminative gains ($+0.0155$ and $+0.0215$ AUROC).
- **Removed Unsupported Claims & Overstatements:**
  - Removed *"clinically ready"* $\to$ Replaced with *"evaluated under cross-hospital deployment"*.
  - Removed *"proves clinical utility"* $\to$ Replaced with *"demonstrates positive utility under the official challenge framework"*.
  - Removed *"state-of-the-art"* $\to$ Replaced with *"strongest discriminative representation among evaluated models"*.
  - Removed *"optimal threshold in general"* $\to$ Replaced with *"prespecified validation threshold achieving peak utility in sensitivity analysis"*.

---

## G. Reviewer-Risk Resolution Mapping (REV-01 to REV-18)

| Concern ID | Reviewer Concern | Final Manuscript Treatment | Resolution Status |
| :--- | :--- | :--- | :---: |
| **REV-01** | Positive Utility vs. Low PPV | PPV ($18.81\%$) and false alert count ($4,333$) reported alongside utility in Abstract & Results. | **RESOLVED** |
| **REV-02** | High AUROC vs. Alarm Fatigue | Section 5.5 added to Discussion detailing alarm fatigue implications of $16.99$ alerts/100 days. | **RESOLVED** |
| **REV-03** | Test-Set Threshold Optimization | Mandatory threshold sensitivity phrasing enforced in Abstract, Results Section 4.4, and Discussion. | **RESOLVED** |
| **REV-04** | Cross-Hospital Dataset Shift | BIDMC ($N=20,336$) $\to$ Emory ($N=20,000$) transfer explicitly defined in Methods Section 3.1 & Table 1. | **RESOLVED** |
| **REV-05** | Baseline Utility Fairness | `—` displayed in Table 1 for non-evaluated baseline utilities with explanatory footnote. | **RESOLVED** |
| **REV-06** | Utility Normalization Meaning | Mathematical derivation of $U_{\text{official}} = +0.655944$ ($65.59\%$ of range between inaction and oracle) explained in Methods & Results. | **RESOLVED** |
| **REV-07** | Prevalence Shift Impact | Section 5.4 added to Discussion explaining prevalence drop ($8.80\%$ to $5.33\%$) depressing PPV. | **RESOLVED** |
| **REV-08** | Probability Calibration | Brier score ($0.015290$) and ECE ($0.018151$) reported in Results Section 4.2. | **RESOLVED** |
| **REV-09** | Missingness & Time Delta Ablation | Factorial ablations ($m = +0.0155$, $\Delta t = +0.0215$ AUROC) presented in Results Section 4.6. | **RESOLVED** |
| **REV-10** | Lead Time & Onset Offset | $t_{\text{sepsis}} = t_{\text{label}} + 6\text{h}$ shift explicitly defined in Methods Section 3.5. | **RESOLVED** |
| **REV-11** | Retrospective Nature | Explicit disclaimer added: positive utility does not equal prospective clinical trial efficacy. | **RESOLVED** |
| **REV-12** | Two-Hospital Scope | Multi-center expansion beyond two academic health systems noted in Limitations Section 5.9. | **RESOLVED** |
| **REV-13** | Observation Workflow Shifts | $\Delta t$ feature encoding framed as mechanism adapting embeddings to sampling rates. | **RESOLVED** |
| **REV-14** | Overclaiming Readiness | All instances of prohibited terms ("clinically ready", "proves", "state-of-the-art") eliminated. | **RESOLVED** |
| **REV-15** | Resampling Unit | Patient stay cluster bootstrap ($B=1,000$) explicitly defined in Methods & Results. | **RESOLVED** |
| **REV-16** | Multi-Seed Stability | $N=6$ seed stability ($AUROC = 0.9609 \pm 0.0016$, $Utility = +0.6559 \pm 0.0020$) reported in Results Section 4.9. | **RESOLVED** |
| **REV-17** | Reproducibility Hashes | SHA256 hashes for checkpoint, prediction arrays, and split manifests provided in Supplementary Material. | **RESOLVED** |
| **REV-18** | Oracle Ceiling Interpretation | Oracle ceiling ($+1.000000$) explicitly labeled as an *infeasible label-informed mathematical reference bound*. | **RESOLVED** |

---

## H. Final Submission Gate Verdict

```text
================================================================================
                       FINAL SUBMISSION GATE VERDICT
================================================================================
   RECOMMENDED VERDICT : READY FOR SUBMISSION
   REASON              : All 4 primary manuscript files (MD and TEX) have been
                         updated, verified against source-of-truth artifacts,
                         audited for numerical consistency, cleaned of overclaims,
                         and committed/pushed to origin/paper-v1.0.
================================================================================
```
