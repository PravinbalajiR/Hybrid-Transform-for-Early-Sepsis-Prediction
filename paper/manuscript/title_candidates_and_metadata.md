# 📌 TITLE CANDIDATES, MANUSCRIPT METADATA & TRIPOD+AI COMPLIANCE

---

## 1. Title Candidates (5 Options)

1. **Option 1 (Recommended & Final Title):**  
   *High Discrimination Does Not Guarantee Clinical Utility: Cross-Hospital Utility Decomposition for Temporal Sepsis Early Warning*
2. **Option 2 (Architectural & Utility Focus):**  
   *Time-Aware Transformer Representations for Early Sepsis Prediction: Benchmarking Cross-Hospital Discrimination and Net Operational Utility*
3. **Option 3 (Operational & Information Focus):**  
   *Quantifying the Discrimination–Utility Disconnect in Clinical Machine Learning: A Multi-Center Sepsis Early Warning Study*
4. **Option 4 (Decision-Theoretic Focus):**  
   *Beyond AUROC: Dual-Bound Utility Decomposition of Clinical Time-Series Models Across Health Systems*
5. **Option 5 (Short & Direct):**  
   *Predictive Discrimination vs. Operational Utility in Cross-Hospital Sepsis Forecasting*

---

## 2. Keywords
`Sepsis Prediction`, `Clinical Decision Support`, `Time-Aware Transformers`, `Cross-Hospital Transportability`, `Clinical Utility Metric`, `Decision-Theoretic Evaluation`, `Alarm Fatigue`, `ICU EHR Modeling`

---

## 3. Primary Contribution Statement

This study makes five primary scientific and methodological contributions:
1. **Utility-Centered Cross-Hospital Evaluation:** We present a rigorous cross-hospital deployment evaluation testing a high-discrimination Time-Aware Transformer ($M3$, AUROC = $0.9617$) under asymmetric decision costs on a held-out test cohort of $20,000$ ICU stays ($753,927$ hourly observations) from Emory University Hospital.
2. **Structured Model Progression ($M1$–$M5$):** We benchmark a controlled model family (XGBoost, Plain Transformer, GRU-D, TCN, Time-Aware Transformer, Organ-Aware Hybrid, Multi-Hybrid MoE), establishing that compact time-aware embeddings provide peak cross-hospital discrimination while further architectural complexity yields no additional gains.
3. **Dual-Bound Utility Decomposition Framework:** We introduce a decision-ceiling framework separating: (i) an infeasible label-informed upper bound (`GROUND_TRUTH_ORACLE_CEILING = +0.8262`), (ii) diagnostic hindsight policy limits (`HINDSIGHT_GRID_SCORE_POLICY_CEILING = -0.1983`), (iii) counterfactual patient-adaptive headroom (`PATIENT_ADAPTIVE_THRESHOLD_CEILING = +0.2819`), and (iv) prespecified deployable performance (`FROZEN_MODEL_UTILITY = -0.2573`).
4. **Leakage-Safe Predictability & Operational Workload Audit:** We quantify operational alert frequency ($16.99$ alerts / 100 patient-days, PPV = $18.81\%$) and evaluate adaptive threshold predictability using a locked pipeline trained strictly on development data, establishing that adaptive policy requirements are not predictable from early features.
5. **Demonstration of the Discrimination–Utility Disconnect:** We provide empirical evidence that evaluating predictive discrimination alone (AUROC/AUPRC) can mask severe operational failure modes in clinical alarm systems.

---

## 4. TRIPOD+AI Compliance Mapping

| TRIPOD+AI Item | Description | Compliance Status | Section Location in Paper |
| :--- | :--- | :---: | :--- |
| **Title & Abstract** | Identify study as ML prediction model development & cross-site validation | **COMPLIANT** | Title & Abstract |
| **Source of Data** | Describe study design, health system sources, and data collection dates | **COMPLIANT** | Methods Section 3.1 & Table 1 |
| **Participants** | State eligibility criteria, exclusion rules, and sample flow | **COMPLIANT** | Methods Section 3.1 & 3.2 |
| **Predictors** | Define all physiological, lab, and demographic predictors used | **COMPLIANT** | Methods Section 3.3 |
| **Outcome** | Define Sepsis-3 outcome criteria, onset window, and timing determination | **COMPLIANT** | Methods Section 3.1 & 3.9 |
| **Sample Size** | Report total patient counts, septic events, and hourly records | **COMPLIANT** | Methods Section 3.1 & Table 1 |
| **Model Specification** | Provide detailed neural network, embedding, and loss function specifications | **COMPLIANT** | Methods Section 3.4 & 3.5 |
| **External Validation** | Evaluate model on an independent external hospital cohort (Set B Emory) | **COMPLIANT** | Results Section 4.1 & 4.4 |
| **Calibration & Discrimination**| Report AUROC, AUPRC, Brier score, ECE, and calibration metrics | **COMPLIANT** | Results Section 4.1 & Table 2 |
| **Clinical Utility & Risk** | Evaluate deployable net utility, decision ceilings, and workload alert burden | **COMPLIANT** | Results Section 4.5, 4.6 & Table 4 |

---

## 5. Reviewer-Risk & Claim Discipline Checklist

- [x] **NO Overclaiming of Causality:** Replaced "proving", "statistically proving", "definitively establishing" with *"consistent with"*, *"provides evidence for"*, *"supports the interpretation that"*, *"within the evaluated setting"*.
- [x] **NO Prohibited Terms:** "State-of-the-art", "proves information limitation", "clinically ready", and "solves sepsis prediction" are strictly prohibited and omitted.
- [x] **Infeasible Upper Bounds Labeled:** `GROUND_TRUTH_ORACLE_CEILING` (+0.826246) and `PATIENT_ADAPTIVE_THRESHOLD_CEILING` (+0.281895) are explicitly labeled as **infeasible label-informed upper bounds**, NOT clinical ceilings.
- [x] **Cohort Attribution Verified:** Verified Set A = BIDMC ($N=20,336$), Set B = Emory ($N=20,000$).
- [x] **Leakage-Safe Predictability:** Predictability model locked on Set A development data and evaluated ONCE on Set B test data ($AUPRC=0.2653$).
