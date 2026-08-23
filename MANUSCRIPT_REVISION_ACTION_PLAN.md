# 📋 MANUSCRIPT REVISION ACTION PLAN (`MANUSCRIPT_REVISION_ACTION_PLAN.md`)

**Repository:** `Hybrid-Transform-for-Early-Sepsis-Prediction`  
**Branch:** `paper-v1.0`  
**Commit Baseline:** `0847adb`  
**Goal:** Complete evidence-based, publication-grade revision plan ensuring 100% manuscript claim discipline, baseline honesty, threshold isolation, and operational workload transparency.

---

## 1. Action Plan Strategy & Scope

This action plan details every specific text, table, figure, and structural refinement required across the manuscript suite (`final_publication_manuscript.md`, `final_publication_manuscript.tex`, `full_research_paper.md`, `full_research_paper.tex`).

**Core Mandates:**
1. **DO NOT retrain any model** or modify frozen checkpoints (`best_m3_frozen.pt`).
2. **DO NOT alter authoritative metrics:** AUROC = `0.961726`, AUPRC = `0.423114`, Utility = `+0.655944`, Accuracy = `0.971542`, Brier = `0.015290`, ECE = `0.018151`.
3. **DO NOT overclaim clinical effectiveness.**
4. **DO NOT assert that $th=0.190$ is globally optimal;** frame it as the prespecified validation threshold that achieved peak utility in sensitivity analysis.
5. **Report baseline utilities honestly;** display `—` for models without preserved hourly prediction arrays.

---

## 2. Itemized Action Items by Section

### Action 1: Abstract Refinement
- **Target Files:** `final_publication_manuscript.md`, `final_publication_manuscript.tex`, `full_research_paper.md`, `full_research_paper.tex`
- **Current Issue:** Abstract reports AUROC (`0.9617`) and Utility (`+0.6559`) but omits exact alert counts and PPV.
- **Required Edit:**
  - Add exact workload metrics to Abstract: *"At prespecified threshold th=0.190, M3 achieved an Official Normalized Utility of +0.6559 (95% CI: [+0.6310, +0.6800]), issuing 16.99 alerts per 100 patient-days (5,337 total alerts, 1,004 true positive, 4,333 false positive; alert PPV = 18.81%)."*
  - Add explicit qualification: *"This utility score reflects performance under the official PhysioNet 2019 decision-cost function and does not constitute direct proof of prospective clinical effectiveness."*

---

### Action 2: Threshold Provenance & Wording Enforcement
- **Target Files:** Methods Section 3.4, Results Section 4.4, Discussion Section 5.1
- **Current Issue:** Risk of reviewer assuming threshold $th=0.190$ was tuned on Emory test data.
- **Required Edit:**
  - Explicitly document 2-stage threshold isolation:  
    1. *"Stage 1 (Validation Selection): Candidate thresholds th in [0.01, 0.99] were evaluated strictly on the BIDMC validation split (N=4,144), identifying th*=0.190 as the utility-maximizing operating point."*  
    2. *"Stage 2 (External Evaluation): The frozen model and locked threshold th=0.190 were evaluated once on the held-out Emory test cohort (N=20,000)."*
  - Enforce mandatory phrasing:  
    *"Among the evaluated thresholds in the sensitivity analysis, the prespecified validation threshold of 0.190 achieved the highest observed official utility on the independent Emory test cohort."*

---

### Action 3: Baseline Utility Table Standardization
- **Target Files:** Table 2 / Table 1 in all manuscript files (`final_publication_manuscript.md`, `.tex`)
- **Current Issue:** Baseline utility column needs strict consistency regarding evaluated vs non-evaluated baselines.
- **Required Edit:**
  - Ensure baseline matrix displays `—` for non-saved prediction baselines:
    ```text
    PhysioNet Challenge Baseline : — (AUROC = 0.8420, AUPRC = 0.2150)
    XGBoost Baseline (M1)        : — (AUROC = 0.8842, AUPRC = 0.2851)
    Plain Transformer (M2)       : — (AUROC = 0.9265, AUPRC = 0.3412)
    GRU-D (Che et al., 2018)     : — (AUROC = 0.9415, AUPRC = 0.3780)
    TCN (Temporal Convolutional) : — (AUROC = 0.9380, AUPRC = 0.3650)
    M3 (Time-Aware Transformer)  : +0.6559 (AUROC = 0.9617, AUPRC = 0.4231)
    ```
  - Add table footnote: *"`—` indicates baselines for which raw hourly prediction arrays were not preserved, preventing official PhysioNet utility calculation without model re-training."*

---

### Action 4: Operational Workload & Alarm Fatigue Section Expansion
- **Target Files:** Results Section 4.3, Discussion Section 5.2
- **Current Issue:** Workload numbers are present in Table 4, but clinical alarm fatigue consequences need deeper discussion.
- **Required Edit:**
  - Expand Section 4.3:
    *"Operational audit across 753,927 hourly records revealed that M3 issued 5,337 total alerts on the Emory test cohort (1,004 true positive sepsis alerts, 4,333 non-sepsis false alerts). This corresponds to an alert frequency of 16.99 alerts per 100 patient-days (0.267 alerts/patient) and an alert PPV of 18.81%. While 74.14% of non-septic patients triggered zero alerts, 25.86% experienced at least one false alert."*
  - Expand Discussion Section 5.2:
    *"An alert PPV of 18.81% highlights the operational challenge of deploying early warning algorithms in ICUs. Although positive utility (+0.6559) indicates that early intervention credit outweighs false alarm penalties under the challenge scoring schema, a rate of 16.99 alerts per 100 patient-days requires integration with clinical triage protocols to mitigate nurse alarm fatigue."*

---

### Action 5: Prohibited Language Clean-Up (Claim Audit)
- **Target Files:** All markdown and TeX manuscript files
- **Required Edit:**
  - Replace *"clinically ready"* $\to$ *"evaluated under cross-hospital deployment"*
  - Replace *"proves clinical utility"* $\to$ *"demonstrates positive utility under the official challenge framework"*
  - Replace *"optimal threshold"* $\to$ *"prespecified threshold achieving peak utility in sensitivity analysis"*
  - Replace *"state-of-the-art performance"* $\to$ *"strongest discriminative representation among evaluated models"*

---

## 3. Implementation Workflow & Verification Steps

1. **Step 1:** Verify consistency across markdown and LaTeX files (`final_publication_manuscript.md`, `.tex`, `full_research_paper.md`, `.tex`).
2. **Step 2:** Execute automated consistency check ([`results/revised_publication/final_consistency_check.txt`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/results/revised_publication/final_consistency_check.txt)).
3. **Step 3:** Confirm git working tree cleanliness (`git status`).

---

## 4. Verification Check

```text
================================================================================
               MANUSCRIPT REVISION ACTION PLAN VERDICT
================================================================================
   PLAN STATUS   : READY FOR EXECUTION
   TARGET BRANCH : paper-v1.0
   SAFETY BOUNDS : ZERO MODEL RETRAINING / ZERO NUMERICAL MUTATION
================================================================================
```
