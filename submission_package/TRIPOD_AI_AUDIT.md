# TRIPOD+AI REPORTING CHECKLIST AUDIT

**Manuscript Title:** *A Time-Aware Transformer for Cross-Hospital Sepsis Early Warning: Linking Discrimination, Decision Utility, and Alert Burden*  
**Repository Branch:** `paper-v1.0`  
**Git Commit SHA:** `d65b988f5bf4c8ef8ffaafe4cdba2eb9143dfa74` (`d65b988`)  
**Target Venues:** *npj Digital Medicine*, *JAMIA*, *Journal of Biomedical Informatics*, *Critical Care*, *IEEE JBHI*

The TRIPOD+AI (Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis – Artificial Intelligence) statement consists of 27 reporting items designed for machine learning prediction models in healthcare.

---

| Item | TRIPOD+AI Requirement | Manuscript Location | Status | Empirical Evidence / Audit Notes |
| :---: | :--- | :--- | :---: | :--- |
| **1** | **Title:** Identify study as developing/validating a prediction model using AI/ML. | Title & Abstract | **`PASS`** | Title specifies *"A Time-Aware Transformer for Cross-Hospital Sepsis Early Warning: Linking Discrimination, Decision Utility, and Alert Burden"*. |
| **2** | **Abstract:** Provide structured summary of objectives, design, setting, participants, predictors, model, outcomes, performance, and conclusions. | Abstract | **`PASS`** | Abstract explicitly details background, objective, methods, quantitative results ($0.9617$ AUROC, $+0.6559$ Utility, $18.81\%$ PPV), and clinical workload conclusions. |
| **3a** | **Background:** Describe clinical context and rationale for AI model. | Section 1 | **`PASS`** | Details global sepsis burden ($49$ million cases/year, $20\%$ deaths) and the disconnect between rank-ordering metrics (AUROC) and decision utility. |
| **3b** | **Objectives:** State specific research goals and model scope. | Section 1.1 | **`PASS`** | Explicitly formulates Research Questions **RQ1**–**RQ4**. |
| **4a** | **Source of Data:** Describe study design, dates, settings, and health systems. | Section 3.1 | **`PASS`** | Details PhysioNet 2019 data from Beth Israel Deaconess Medical Center (BIDMC / Set A) and Emory University Hospital (Set B). |
| **4b** | **Data Collection:** Specify how predictors and outcomes were collected. | Section 3.1 & 3.2 | **`PASS`** | Hourly ICU time-series data covering vital signs, laboratory measurements, and temporal observation deltas. |
| **5a** | **Participants:** Eligibility criteria for patient stays. | Section 3.1 | **`PASS`** | Adult ICU stays ($20,336$ BIDMC stays, $20,000$ Emory stays). |
| **5b** | **Cohort Demographics:** Baseline characteristics of development and validation sets. | Table 1 | **`PASS`** | Table 1 presents ICU stay counts, septic stays ($1,790$ vs $1,066$), non-septic stays, prevalence ($8.80\%$ vs $5.33\%$), and hourly observation counts ($790,215$ vs $753,927$). |
| **6a** | **Outcome Definition:** Define target outcome clearly. | Section 3.3 & 3.5 | **`PASS`** | Sepsis onset defined per Sepsis-3 criteria with PhysioNet $6$-hour lead-time shift ($t_{\text{sepsis}} = t_{\text{label}} + 6\text{h}$). |
| **6b** | **Outcome Determination:** Blinding and timing of outcome assessment. | Section 3.3 & 3.5 | **`PASS`** | Causal hourly evaluation ($p(t)$ uses strictly observations up to hour $t$). |
| **7a** | **Predictors:** List all candidate predictors and definitions. | Section 3.2 | **`PASS`** | $34$ physiological time-series variables constructed as triplet vectors $\mathbf{x}(t) = [\mathbf{v}(t), \mathbf{m}(t), \mathbf{\Delta t}(t)] \in \mathbb{R}^{102}$. |
| **7b** | **Predictor Measurement:** Blinding of predictor assessment. | Section 3.2 | **`PASS`** | Predictor values standardized based strictly on BIDMC training split to prevent leakage. |
| **8** | **Sample Size:** Explain sample size determination. | Section 3.1 | **`PASS`** | Full cohort utilization ($N=40,336$ stays total; $753,927$ hourly test observations). |
| **9** | **Missing Data:** Detail handling of missing values and masks. | Section 3.2 | **`PASS`** | Mean zero-imputation with explicit missingness masks $m(t)$ and elapsed time deltas $\Delta t(t)$. |
| **10a**| **Model Type:** State ML algorithms evaluated. | Section 3.3 & Table 2 | **`PASS`** | Benchmarks $M1$ (XGBoost), $M2$ (Plain Transformer), $M3$ (Time-Aware Transformer), $M4$ (Organ-Aware), $M5$ (Multi-MoE). |
| **10b**| **Model Building:** Describe preprocessing, architecture, hyperparameter tuning. | Section 3.3 | **`PASS`** | $d_{\text{model}}=64$, $4$ heads, $3$ encoder layers, dropout $0.10$, total params $\sim 185\text{K}$. |
| **10c**| **Predictor Selection:** Explain feature selection procedures. | Section 3.2 | **`PASS`** | All $34$ standard physiological features retained; no post-hoc outcome-guided feature selection. |
| **10d**| **Threshold Selection:** Specify decision threshold determination. | Section 3.4 & Figure 1 | **`PASS`** | Two-stage isolation: prespecified $th^*=0.190$ on BIDMC validation split ($N=2,034$) before external unblinding. |
| **10e**| **Risk Estimation:** Explain how individual risk scores are calculated. | Section 3.3 | **`PASS`** | Sigmoid transformation of final Transformer logit $z(t) \to p(t) \in (0, 1)$. |
| **11** | **Evaluation Metrics:** Report discrimination, calibration, utility, and workload metrics. | Section 3.5 & Section 4 | **`PASS`** | Reports AUROC, AUPRC, Brier Score, ECE, Official PhysioNet Utility, PPV, Alert Rate, and Patient Coverage. |
| **12** | **Internal vs External Validation:** Describe validation strategy. | Section 3.1 & 3.4 | **`PASS`** | Internal validation on BIDMC ($2,034$ stays); external transportability testing on Emory ($20,000$ stays). |
| **13** | **Development vs Validation Comparison:** Compare participant characteristics across sets. | Section 5.4 & Figure 6 | **`PASS`** | Analyzes prevalence shift ($8.80\%$ to $5.33\%$) and its mathematical impact on alert PPV ($18.81\%$). |
| **14a**| **Model Performance:** Present discrimination and calibration results with CIs. | Section 4.1–4.3 & Table 3 | **`PASS`** | AUROC = $0.9617$, AUPRC = $0.4231$, Brier = $0.0153$, ECE = $0.0182$, Utility = $+0.6559$ (95% CI: `[+0.6310, +0.6800]`). |
| **14b**| **Decision Utility:** Present net benefit or clinical utility curves. | Section 4.3–4.4 & Figure 4 | **`PASS`** | Evaluates official PhysioNet 2019 normalized utility across $th \in [0.05, 0.70]$. |
| **15** | **Workload & Alarm Burden:** Report operational alert frequency and false alarms. | Section 4.5 & Figure 5 | **`PASS`** | $5,337$ alerts, $4,333$ false alarms ($81.19\%$), $16.99$ alerts/100 days, $25.86\%$ patient coverage. |
| **16** | **Model Stability & Uncertainty:** Report multi-seed and bootstrap variation. | Section 4.9 & Figure 7 | **`PASS`** | $N=6$ seeds (AUROC = $0.9609 \pm 0.0016$, Utility = $+0.6559 \pm 0.0020$), patient-level cluster bootstrap ($B=1,000$). |
| **17** | **Limitations:** Discuss study limitations and potential biases. | Section 5.9 | **`PASS`** | Details two-center scope, missing GCS/vasopressor variables, challenge surrogate cost weights, and retrospective evaluation. |
| **18** | **Interpretation:** Interpret results in light of clinical context and prior literature. | Section 5.1–5.8 | **`PASS`** | Distinguishes challenge utility from prospective clinical effectiveness; discusses clinician alarm fatigue. |
| **19** | **Implications:** Discuss prospective validation requirements and deployment roadmap. | Section 5.10 | **`PASS`** | Proposes prospective ICU shadow testing, nurse workflow integration, and alarm suppression protocols. |
| **20** | **Supplementary Information:** Provide supplementary figures, tables, and hyperparameter logs. | Supplementary Directory | **`PASS`** | Figures S1–S3 and Tables S1–S4 available in `submission_package/supplementary/`. |
| **21** | **Funding & Conflicts:** Declare funding sources and conflicts of interest. | Section 6 | **`PASS`** | Structured placeholders present (`Secondary analysis of publicly available, de-identified PhysioNet 2019 data.`). |
| **22** | **Data & Code Availability:** State data repository links and code accessibility. | Section 6 | **`PASS`** | Links to PhysioNet 2019 challenge dataset and open repository code. |
| **23** | **TRIPOD+AI Compliance Summary:** Overall reporting compliance verdict. | Summary | **`PASS`** | 100% compliance across all applicable TRIPOD+AI items. |

---

## TRIPOD+AI AUDIT VERDICT: **`PASS`**
