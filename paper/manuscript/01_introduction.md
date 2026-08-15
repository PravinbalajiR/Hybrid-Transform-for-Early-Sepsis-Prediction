# Section 1: Introduction

## 1.1 Clinical Problem: Sepsis Severity & Time Sensitivity
Sepsis is a life-threatening organ dysfunction caused by a dysregulated host response to infection (Singer et al., 2016). It represents a primary cause of mortality in Intensive Care Units (ICUs) worldwide, accounting for millions of deaths annually and imposing a severe financial burden on global healthcare systems. In clinical management, sepsis is time-sensitive: each hour of delayed administration of effective antimicrobial therapy following the onset of septic shock increases patient mortality by approximately 4% to 8% (Kumar et al., 2006; Seymour et al., 2017). Consequently, automated early warning systems capable of identifying sepsis hours prior to overt clinical onset are of interest for supporting timely physiological resuscitation.

## 1.2 The ICU Data Challenge: Irregularity and Informative Missingness
Despite the need for automated risk scoring, real-world Electronic Health Record (EHR) time-series data present substantial analytical hurdles. In intensive care settings, physiological vital signs (e.g., heart rate, blood pressure, oxygen saturation) are measured continuously or at frequent hourly intervals, whereas laboratory measurements (e.g., lactate, white blood cell count, creatinine) are ordered irregularly based on clinical suspicion. This results in two distinct characteristics of ICU time-series:
1. **Irregular Sampling Intervals ($\Delta t$):** The elapsed time between consecutive observations varies widely across different physiological features and individual patients.
2. **Informative Missingness ($\mathbf{m}$):** Observations are unobserved not at random, but as a reflection of clinical decision-making; the presence, frequency, and pattern of a test order may encode signals regarding a patient's underlying physiological trajectory (Che et al., 2018; Rubin et al., 2018).

Standard statistical modeling techniques often obscure these signals by applying aggressive global imputation (e.g., last observation carried forward or mean imputation) or assuming uniform time steps, discarding information inherent in observation timing and missingness patterns.

## 1.3 Evolution of Predictive Approaches
Over the past decade, predictive modeling for early sepsis identification has evolved through three primary paradigms:
- **Traditional Machine Learning:** Initial approaches relied on static scoring systems or feature-engineered tree ensembles, such as InSight (Desautels et al., 2016) and XGBoost baselines (Zabihi et al., 2020). While computationally efficient, these models flatten temporal dynamics or rely on fixed summary windows, limiting their ability to model continuous physiological trajectories.
- **Sequential Deep Learning:** Recurrent Neural Networks (RNNs), Long Short-Term Memory networks (LSTMs), and Gated Recurrent Units with Decay (GRU-D) (Lipton et al., 2016; Che et al., 2018) introduced step-by-step sequential processing to capture temporal dependencies. However, recurrent architectures suffer from sequential processing bottlenecks, vanishing gradients over long horizons, and difficulty in modeling non-linear, multi-scale temporal gaps.
- **Transformer-Based Models:** Applications of Transformer encoders (Vaswani et al., 2017) leverage self-attention mechanisms to learn direct pairwise dependencies across sequence length without step-wise recurrence (Horn et al., 2020; Tipirneni and Reddy, 2022). However, many existing clinical Transformer adaptations treat ICU data as regularly spaced sequence steps or rely on static positional encodings, leaving the explicit interaction between continuous temporal gaps and observation patterns insufficiently explored.

## 1.4 Research Gap
Many existing clinical predictive models do not explicitly represent both irregular observation timing ($\Delta t$) and observation patterns ($\mathbf{m}$) directly within the neural input representation. Furthermore, empirical evidence is needed to evaluate how explicit temporal representations influence predictive discrimination ($\text{AUROC}/\text{AUPRC}$), calibration ($\text{ECE}$), and operational alerting performance under standardized challenge utility scoring.

## 1.5 Study Overview & Experimental Design
To address this gap, this study presents a controlled experimental investigation evaluating how explicit representations of physiological values, observation patterns, and temporal information influence early sepsis prediction. Rather than evaluating a single isolated model, we establish a leak-free benchmark progression evaluated on the PhysioNet 2019 ICU dataset ($N = 40,336$ patients):
1. **M1 (XGBoost Baseline):** Standard tree ensemble operating on imputed dynamic summary windows.
2. **M2 (Plain Transformer):** 3-Layer Causal Transformer Encoder operating on naive imputed vital values.
3. **M3 (Time-Aware Transformer):** Primary model adapting continuous frequency temporal embeddings (Time2Vec; Kazemi et al., 2019) and missingness masks into a unified Transformer.
4. **M3 Component Ablations:** Systematic ablation variants (M3-Full, M3-No-Time, M3-No-Mask, M3-No-Time-No-Mask) quantifying the isolated contributions of time deltas and observation masks.
5. **M4 & M5 Architectural Explorations:** Exploratory multi-branch hybrid architectures incorporating organ subsystem token injection (M4) and Mixture-of-Experts routing (M5).

All models were trained, validated, and evaluated under strict experimental control: zero patient overlap across splits, Z-score normalization fit strictly on training data, decision thresholds pre-specified using validation data only, and single-pass evaluation on the held-out test cohort ($N = 20,000$).

## 1.6 Principal Contributions
The principal contributions of this work are fourfold:
1. **Time-Aware Input Representation:** We demonstrate a compact Transformer framework that explicitly projects physiological values, missingness masks, and continuous frequency time-delta representations (adapting Time2Vec) into a unified self-attention space, achieving an AUROC of 0.9617 and an AUPRC of 0.4231 on the held-out test cohort (an improvement of +0.0352 AUROC and +0.0691 AUPRC over the values-only Transformer baseline M2).
2. **Controlled Component Ablation:** We perform a leak-free ablation study demonstrating that incorporating continuous temporal embeddings extends early warning lead time (+1.0 hour over M2, reaching 5.2h in M3-Time+Delta and 6.2h in M3-Full at validation-optimal operating threshold $th_{\text{val\_opt}}=0.44$), while missingness masks enhance precision (+0.0449 PPV when added to Time+Delta).
3. **Validation-Locked Operating Protocol:** We evaluate a strict pre-specified operating protocol where decision thresholds are selected exclusively on validation data ($th_{\text{val\_opt}}=0.44$), followed by single-pass test evaluation and patient-level 1,000 bootstrap confidence interval analysis.
4. **Operational Utility Insights:** We report a transparent patient-level decomposition illustrating that while time-aware representations substantially improve predictive discrimination and calibration ($\text{ECE} = 0.0407$), strong discrimination does not automatically translate into positive operational utility under raw hourly alerting protocols, exposing a critical gap between discrimination metrics and challenge utility scoring.
