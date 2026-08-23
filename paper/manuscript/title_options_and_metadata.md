# 📌 MANUSCRIPT TITLES, METADATA & CONTRIBUTION STATEMENTS

---

## 1. Title Alternatives (10 Options)

1. **Primary Recommendation (Final Title):**  
   *A Time-Aware Transformer for Cross-Hospital Sepsis Early Warning: From Discriminative Performance to Decision Utility and Alert Burden*
2. **Methodological Focus:**  
   *Dual-Bound Utility Decomposition for Clinical AI: Diagnosing Information and Policy Limits in Sepsis Prediction*
3. **Evaluation Focus:**  
   *Beyond AUROC: Evaluating Asymmetric Cost Utility and Cross-Hospital Transfer of Time-Aware Transformers for Early Sepsis Prediction*
4. **Clinical AI / Machine Learning Focus:**  
   *Quantifying the Information-Utility Gap in Cross-Hospital Sepsis Prediction Using Time-Aware Transformers*
5. **Short & Direct:**  
   *Discriminative Discrimination vs. Clinical Utility in Cross-Hospital Sepsis Forecasting*
6. **Information-Theoretic Focus:**  
   *Information Limits of Scalar Risk Scores in Clinical Early Warning Systems: A Multi-Center Utility Analysis*
7. **Cross-Hospital Transfer Focus:**  
   *Cross-Hospital Transferability and Net Utility Breakdown of Deep Clinical Time-Series Models*
8. **Decision-Theoretic Focus:**  
   *Decision-Theoretic Evaluation of ICU Sepsis Models: Dissecting the Discrimination–Utility Disconnect*
9. **Empirical Forensic Focus:**  
   *Why High-AUROC Models Can Fail in Deployment: A Multi-Center Oracle Decomposition of Clinical Utility*
10. **Journal / Comprehensive Title:**  
    *Utility Decomposition of Temporal Deep Learning Models for Sepsis Prediction: Evaluating Cross-Hospital Generalization Under Asymmetric Decision Costs*

---

## 2. Keywords
`Sepsis Prediction`, `Clinical Decision Support`, `Time-Aware Transformers`, `Cross-Hospital Generalization`, `Clinical Utility Metric`, `Decision-Theoretic Evaluation`, `Information Gap`, `Machine Learning in ICU`

---

## 3. Main Contribution Statement

This study makes five primary scientific and methodological contributions:
1. **Utility-Centered Cross-Hospital Evaluation:** We present an empirical cross-hospital deployment analysis evaluating a high-discrimination Time-Aware Transformer model ($M3$, AUROC = $0.9617$) under asymmetric decision costs on a held-out cohort of $20,000$ ICU stays ($753,927$ hourly observations).
2. **Dual-Bound Utility Decomposition Framework:** We formulate a decision-ceiling framework separating: (i) perfect-information decision potential (`GROUND_TRUTH_ORACLE_CEILING`), (ii) diagnostic hindsight policy limits (`HINDSIGHT_GRID_SCORE_POLICY_CEILING`), (iii) counterfactual patient-adaptive headroom (`PATIENT_ADAPTIVE_THRESHOLD_CEILING`), and (iv) prespecified deployable performance (`FROZEN_MODEL_UTILITY`).
3. **Quantification of the Information/Representation Gap:** We provide statistical proof ($\Delta = +1.0246, 95\%\text{ CI: }[+0.9997, +1.0494], p < 0.0001$) of a substantial separation between perfect-information decision making and observable score representations under a global policy.
4. **Patient-Level Uncertainty & Multi-Seed Stability:** We conduct patient-level bootstrap resampling ($B=1,000$) and multi-seed stability checks across $6$ distinct initialization seeds, establishing that the negative deployable utility is robust ($\text{std} \le 0.0020$) and not an artifact of random sampling or initialization.
5. **Demonstration of the Discrimination–Utility Disconnect:** We provide empirical evidence that evaluating predictive discrimination alone (AUROC/AUPRC) can mask severe operational failure modes in clinical alarm systems.

---

## 4. Reproducibility Statement
All code, preprocessed data manifests, model weights, and prediction artifacts are frozen on Git branch `paper-v1.0` (Commit `c3eb504`). All primary point estimates and confidence intervals were verified independently with zero arithmetic discrepancy ($\le 10^{-10}$). Random seeds are fixed (`seed = 42`), and all execution scripts are fully deterministic.

---

## 5. Data & Code Availability Statement
- **Data Availability:** The PhysioNet/Computing in Cardiology Challenge 2019 dataset (Set A and Set B) is publicly available on PhysioNet (`https://physionet.org/content/challenge-2019/1.0.0/`).
- **Code Availability:** All preprocessing pipelines, time-aware transformer model definitions, evaluation scripts, and independent verification modules are available in the GitHub repository: `https://github.com/PravinbalajiR/Hybrid-Transform-for-Early-Sepsis-Prediction`.
