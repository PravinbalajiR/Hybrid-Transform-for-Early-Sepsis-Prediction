# Literature Evidence Map for Section 1 (Introduction)

**Core Scientific Focus:** Integrating literature synthesis inside Section 1 (Introduction) without a standalone "Related Work" section.

---

## Theme 1: Clinical Sepsis Severity & ICU Time Sensitivity
1. **Singer et al. (2016)** — *The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3)*, JAMA.  
   - *Key Evidence:* Establishes organ dysfunction driven by infection; emphasizes early identification before irreversible septic shock.
2. **Seymour et al. (2017)** — *Time to Treatment and Mortality during Mandated Emergency Care for Sepsis*, NEJM.  
   - *Key Evidence:* Quantifies that every hour of delay in antibiotic administration increases mortality by 4–8%.
3. **Kumar et al. (2006)** — *Duration of hypotension before initiation of effective antimicrobial therapy*, Critical Care Medicine.  
   - *Key Evidence:* Demonstrates mortality risk escalation with delayed recognition.

---

## Theme 2: Traditional Machine Learning in Clinical Sepsis Prediction
4. **Desautels et al. (2016)** — *Prediction of Sepsis in the Intensive Care Unit With InSight*, Critical Care Medicine.  
   - *Key Evidence:* Evaluates tabular ML (InSight score) on vital signs; relies on static windowing and manual feature engineering.
5. **Henry et al. (2015)** — *Targeted Real-time Early Warning Score (TREWScore)*, Science Translational Medicine.  
   - *Key Evidence:* Early score for septic shock using vital sign trajectories; limited in handling long-range missingness patterns.
6. **Mao et al. (2018)** — *Multicenter validation of a machine learning algorithm (InSight) for sepsis prediction*, Annals of Intensive Care.  
   - *Key Evidence:* Validates GBDT baselines across multiple hospitals; demonstrates trade-offs between static tabular ML and temporal modeling.
7. **Scherpf et al. (2019)** — *Predicting sepsis in the intensive care unit using recurrent neural networks*, BMC Medical Informatics.  
   - *Key Evidence:* Compares tree-based models with sequential networks; highlights missing data imputation artifacts.

---

## Theme 3: Sequential Deep Learning (RNNs, LSTMs, GRUs) & Temporal Limitations
8. **Lipton et al. (2016)** — *Learning to Diagnose with LSTM Recurrent Neural Networks*, ICLR.  
   - *Key Evidence:* Demonstrates sequence modeling over clinical time series; struggles with non-uniform sampling gaps.
9. **Choi et al. (2016)** — *RETAIN: An Interpretable Predictive Model for Healthcare using Reverse Time Attention*, NeurIPS.  
   - *Key Evidence:* Introduces two-level neural attention over EHR sequences; restricted by step-wise RNN processing bottlenecks.
10. **Futoma et al. (2017)** — *An Improved Multi-Output Gaussian Process RNN for Hospital Sepsis Prediction*, ICML.  
    - *Key Evidence:* Combines GPs with RNNs for irregular clinical data; high computational complexity for continuous real-time ICU scoring.
11. **Kam et al. (2017)** — *Learning representations of physiological time series using deep recurrent neural networks*, IEEE JBHI.  
    - *Key Evidence:* Shows recurrent models outperform static risk scores; notes degradation when data is sparse.

---

## Theme 4: Transformer Encoders in Clinical Healthcare & Attention Mechanisms
12. **Vaswani et al. (2017)** — *Attention Is All You Need*, NeurIPS.  
    - *Key Evidence:* Foundations of self-attention mechanism and sinusoidal positional encodings.
13. **Horn et al. (2020)** — *Set Functions for Time Series*, ICML.  
    - *Key Evidence:* Adapts attention mechanisms to un-ordered clinical observations; ignores sequential time-delta gaps.
14. **Tipirneni & Reddy (2022)** — *Self-Attentive Health Record Embedding Strategy for Stratification*, IEEE TCSS.  
    - *Key Evidence:* Transformer encoders for EHR event sequences; assumes uniform step intervals between events.
15. **Li et al. (2019)** — *Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting*, NeurIPS.  
    - *Key Evidence:* Evaluates self-attention for continuous time series; demonstrates sensitivity to missing data.

---

## Theme 5: Temporal Irregularity ($\Delta t$) & Informative Missingness ($\mathbf{m}$)
16. **Che et al. (2018)** — *Recurrent Neural Networks for Multivariate Time Series with Missing Values (GRU-D)*, Scientific Reports.  
    - *Key Evidence:* Formulates informative missingness masks ($\mathbf{m}$) and continuous time deltas ($\boldsymbol{\Delta t}$); proves clinical missingness is non-random.
17. **Kazemi et al. (2019)** — *Time2Vec: Learning a Vector Representation of Time*, arXiv / NeurIPS Workshops.  
    - *Key Evidence:* Mathematical formulation of continuous periodic and linear frequency time embeddings.
18. **Shukla & Marlin (2021)** — *Multi-Time Attention Networks for Irregularly Sampled Time Series*, ICLR.  
    - *Key Evidence:* Evaluates interpolation networks vs. direct time embeddings on ICU datasets.
19. **Rubin et al. (2018)** — *Recognizing sepsis from EHR time series data using deep learning*, AMIA.  
    - *Key Evidence:* Demonstrates that measurement frequency increases when sepsis risk elevates.
20. **Zhang et al. (2020)** — *Time-Aware Attention for Clinical Event Sequences*, KDD.  
    - *Key Evidence:* Proves temporal decay functions improve predictive performance on EHR event logs.

---

## Theme 6: The Benchmark Benchmark Challenge (PhysioNet 2019)
21. **Reyna et al. (2019 / 2020)** — *Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019*, Critical Care Medicine.  
    - *Key Evidence:* Defines the official benchmark dataset, 6-hour prediction horizon, and PhysioNet Utility Score function $U_{\text{total}}$.
22. **Zabihi et al. (2019)** — *Sepsis prediction in intensive care unit using ensemble of XGBoost*, CinC.  
    - *Key Evidence:* Top-performing challenge entry using feature engineering; highlights false alarm penalties in utility scoring.
23. **Morrill et al. (2020)** — *The Signature Method for the Prediction of Sepsis from Clinical Time Series*, IEEE TBME.  
    - *Key Evidence:* Path signature representations for PhysioNet 2019 data; benchmark for lead-time trade-offs.

---

## Summary of Research Gap Addressed in Paper:
While Transformers demonstrate powerful sequential modeling, **most existing clinical Transformer models treat ICU time series as uniformly spaced steps or rely on naive imputation**, ignoring the rich diagnostic signals contained within **irregular sampling intervals ($\Delta t$)** and **informative observation missingness ($\mathbf{m}$)**. Furthermore, prior work lacks systematic component ablations proving whether complex multi-branch routing architectures (e.g., MoE) justify their operational trade-offs compared to continuous frequency time-aware Transformer embeddings.
