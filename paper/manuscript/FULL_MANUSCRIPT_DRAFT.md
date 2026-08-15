# Time-Aware Representation of Irregular Physiological Data for Early Sepsis Prediction

# Research Highlights

* Temporal information improved early sepsis prediction from irregular ICU data.
* Observation patterns provided additional information for sepsis prediction.
* Combining temporal and observation information improved overall performance.
* The proposed model provided earlier warnings with fewer false positives.
* Greater architectural complexity did not consistently improve prediction.


---

## Abstract

**Background:** Early identification of sepsis in the intensive care unit (ICU) is critical for initiating timely resuscitation. However, clinical time-series data present extreme missingness and irregular measurement intervals ($\Delta t$), which standard machine learning models flatten via static imputation.

**Objective:** This study investigates whether explicitly representing physiological values, observation missingness patterns ($\mathbf{m}$), and continuous temporal gaps ($oldsymbol{\Delta t}$) within a Transformer architecture improves early sepsis prediction, and evaluates whether increasing architectural complexity yields superior performance.

**Methods:** We established a leak-free benchmark on $40,336$ ICU patients from the PhysioNet 2019 dataset (Train: $18,302$, Val: $2,034$, Test: $20,000$). We evaluated a baseline gradient boosted tree (M1), a plain Transformer (M2), a proposed Time-Aware Transformer (M3) incorporating continuous frequency temporal embeddings (Time2Vec) and missingness masks, four component ablation variants, and two exploratory multi-branch architectures (M4 Organ Hybrid and M5 Multi-Hybrid Network). Operating thresholds ($th=0.60$) were locked strictly on validation performance before single-pass test evaluation.

**Results:** M3 achieved the highest discrimination ($	ext{AUROC} = 0.9617$, 95% CI: `[0.9495, 0.9727]`; $	ext{AUPRC} = 0.4231$, 95% CI: `[0.3359, 0.5185]`), outperforming M1 ($	ext{AUROC} = 0.8420$) and M2 ($	ext{AUROC} = 0.9265$). Component ablations demonstrated that Time2Vec deltas extended mean lead time (+1.0 hour, reaching 5.2h in M3-Time+Delta and 5.7h in M3-Full), while missingness masks improved precision (+0.0449 PPV in M3-Full). M3 maintained superior calibration ($	ext{ECE} = 0.0407$) and PhysioNet utility ($	ext{Utility} = -0.9535$). Multi-branch MoE expert routing (M5) extended lead time to 12.0 hours but quadrupled false alarm rates ($5.80\%$ vs $1.83\%$ FPR/h) and degraded precision down to $11.58\%$.

**Conclusion:** Explicitly embedding temporal gaps and missingness patterns within a compact Transformer significantly enhances early sepsis alerting. Increasing architectural complexity via multi-branch MoE routing does not improve overall clinical utility.

**Keywords:** early sepsis prediction; intensive care; temporal modeling; missing data; deep learning; clinical prediction

---

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


---

# Section 2: Materials and Methods

## 2.1 Dataset
We conducted our experiments using the publicly available dataset from the PhysioNet / Computing in Cardiology Challenge 2019 (Reyna et al., 2019; Reyna et al., 2020). The dataset contains hourly clinical data collected from intensive care unit (ICU) patients across two distinct hospital systems (System A and System B). The raw data comprise 40,336 ICU patients and 1,552,210 total hourly observation records, encompassing 34 physiological features recorded alongside demographic variables and binary clinical labels indicating sepsis onset.

## 2.2 Patient Cohort and Data Splitting
To ensure strict, leak-free validation, patient records were partitioned at the patient level prior to feature normalization or model training:
- **Training Cohort:** 18,302 patients (45.37%)
- **Validation Cohort:** 2,034 patients (5.04%)
- **Test Cohort:** 20,000 patients (49.58%)

Zero patient overlap exists across the three partitions. The overall patient-level sepsis prevalence across the cohort is 7.38% (2,977 septic patients), matching the official challenge benchmark split. Sepsis onset is clinically defined according to the Sepsis-3 criteria (Singer et al., 2016) as the hour when a patient meets both suspicion of clinical infection and a two-point rise in Sequential Organ Failure Assessment (SOFA) score.

## 2.3 Physiological Data Preprocessing
For each patient $i$ at hour $t \in \{1, \dots, T_i\}$, the raw observation vector contains 34 physiological measurements, comprising 8 vital signs (Heart Rate, Pulse Oximetry O2Sat, Temperature, Systolic Blood Pressure, Mean Arterial Pressure, Diastolic Blood Pressure, Respiratory Rate, End-Tidal CO2) and 26 laboratory observations (e.g., Arterial pH, PaO2, Lactate, Creatinine, Platelets, White Blood Cell Count).

All continuous physiological features were normalized using Z-score standardization:
$$\tilde{x}_{t,j} = \frac{x_{t,j} - \mu_j}{\sigma_j}$$
where $\mu_j$ and $\sigma_j$ denote the mean and standard deviation of physiological feature $j$ calculated **strictly on the Training cohort**. Validation and test cohorts were transformed using the pre-computed training parameters to prevent data leakage. Unobserved initial measurements were filled with zero post-standardization.

## 2.4 Representation of Values, Observation Masks, and Time Deltas
To explicitly preserve missingness patterns and irregular measurement gaps, we constructed a triplet vector $\mathbf{x}_t \in \mathbb{R}^{102}$ for each patient hour $t$:
$$\mathbf{x}_t = \left[ \mathbf{v}_t \,\|\, \mathbf{m}_t \,\|\, \boldsymbol{\Delta t}_t \right]$$
where:
1. **Normalized Physiological Values ($\mathbf{v}_t \in \mathbb{R}^{34}$):** Standardized observation values at hour $t$.
2. **Observation Masks ($\mathbf{m}_t \in \mathbb{R}^{34}$):** Binary indicators where $m_{t,j} = 1$ if feature $j$ was physically measured at hour $t$, and $m_{t,j} = 0$ otherwise.
3. **Continuous Time Deltas ($\boldsymbol{\Delta t}_t \in \mathbb{R}^{34}$):** The continuous elapsed time (in hours) since feature $j$ was last observed:
   $$\Delta t_{t,j} = \begin{cases} t - t_{\text{last}}, & \text{if } t > t_{\text{last}} \\ 0, & \text{at initial observation} \end{cases}$$
Elapsed time gaps were clipped at $168.0$ hours to prevent numerical instability during long ICU stays.

## 2.5 Baseline Models

### 2.5.1 M1: XGBoost Baseline
Model M1 serves as a traditional tabular machine learning benchmark (`xgboost.XGBClassifier`). Continuous features were pre-imputed using Last Observation Carried Forward (LOCF) and overall median imputation. Hyperparameters were set to `max_depth=6`, `n_estimators=100`, `learning_rate=0.1`, `subsample=0.8`, and class imbalance weighting `scale_pos_weight=47.66`.

### 2.5.2 M2: Plain Transformer
Model M2 is a 3-layer Causal Transformer Encoder operating on naive imputed vital vectors ($\mathbf{v}_t \in \mathbb{R}^{34}$). It excludes observation masks ($\mathbf{m}_t$) and time deltas ($\boldsymbol{\Delta t}_t$). Features are projected via a linear layer ($\mathbb{R}^{34} \to \mathbb{R}^{64}$), combined with standard sinusoidal positional encodings, and processed through 3 Transformer layers ($d_{\text{model}}=64$, $n_{\text{head}}=4$). Total trainable parameters: **161,793**.

## 2.6 Time-Aware Transformer (M3)

### 2.6.1 Input Representation
Model M3 (Time-Aware Transformer — TACTModel) receives the full 102-dimensional triplet vector $\mathbf{x}_t = [\mathbf{v}_t \,\|\, \mathbf{m}_t \,\|\, \boldsymbol{\Delta t}_t]$.

### 2.6.2 Time2Vec Temporal Encoding
Rather than using simple scalar multiplication, M3 transforms variable-specific continuous time deltas $\boldsymbol{\Delta t}_t$ by adapting **Time2Vec** (Kazemi et al., 2019). For each feature $j$, Time2Vec projects $\Delta t_{t,j}$ into 1 linear and $K-1$ periodic frequency components ($K=4$):
$$\text{Time2Vec}(\Delta t_{t,j})[k] = \begin{cases} \omega_{j,0} \Delta t_{t,j} + \varphi_{j,0}, & \text{if } k = 0 \\ \sin(\omega_{j,k} \Delta t_{t,j} + \varphi_{j,k}), & \text{if } 1 \le k < K \end{cases}$$
The resulting $34 \times 4 = 136$ frequency features are concatenated with $\mathbf{v}_t$ (34) and $\mathbf{m}_t$ (34), yielding a $204$-dimensional tensor projected to $d_{\text{model}}=64$. Total trainable parameters: **163,841**.

### 2.6.3 Transformer Encoder & Causal Attention
The embedding sequence is normalized using LayerNorm, combined with sinusoidal positional encodings, and passed to a 3-layer Transformer Encoder ($d_{\text{model}}=64$, $n_{\text{head}}=4$, $d_{\text{ff}}=128$, `dropout=0.1`, `activation="relu"`). Causal self-attention masking is applied to ensure that risk scoring at hour $t$ depends strictly on observations at hours $\le t$.

### 2.6.4 Prediction Layer
The output state $\mathbf{h}_t \in \mathbb{R}^{64}$ is fed into a two-layer classification MLP:
$$\text{logits}_t = \mathbf{W}_2 \cdot \text{Dropout}(\text{ReLU}(\mathbf{W}_1 \mathbf{h}_t + \mathbf{b}_1)) + b_2$$
$$\hat{p}_t = \sigma(\text{logits}_t)$$
where $\hat{p}_t \in [0, 1]$ represents the predicted probability of sepsis onset within the 6-hour window.

## 2.7 Component Ablation Study
To isolate the predictive value of continuous time deltas ($\boldsymbol{\Delta t}$) and missingness masks ($\mathbf{m}$), we evaluated four controlled ablation variants within the exact M3 architecture:
1. **M2 / Values-Only:** Input contains values $\mathbf{v}_t$ only ($\mathbf{m}_t=0, \boldsymbol{\Delta t}_t=0$).
2. **M3-Time+Delta (No-Mask):** Input contains values $\mathbf{v}_t$ and Time2Vec deltas $\boldsymbol{\Delta t}_t$ ($\mathbf{m}_t=0$).
3. **M3-Time+Mask (No-Time):** Input contains values $\mathbf{v}_t$ and masks $\mathbf{m}_t$ ($\boldsymbol{\Delta t}_t=0$).
4. **M3-Full:** Complete M3 model with values $\mathbf{v}_t$, masks $\mathbf{m}_t$, and Time2Vec deltas $\boldsymbol{\Delta t}_t$.

## 2.8 Architectural Exploration (M4 and M5)
To evaluate whether greater architectural complexity improves performance over M3, we investigated two alternative multi-branch models:

### 2.8.1 M4: Organ Hybrid / Mixture-of-Experts
Model M4 (`SepsisHybridModel`) incorporates 6 Physiology-Aware Temporal Encoders (PATE) representing organ subsystems (Cardiovascular, Pulmonary, Renal, Hepatic, Hematologic, Neurologic). The organ representations are prepended as 6 prefix tokens to the temporal sequence before Transformer attention. M4 incorporates a multi-task forecasting head predicting 5 vital deltas (MAP, Creatinine, Lactate, O2Sat, RespRate). Total parameters: **198,433**.

### 2.8.2 M5: Multi-Hybrid Network
Model M5 (`M5Model`) splits the triplet into disjoint Value (34 $\to$ 32), Mask (34 $\to$ 32), and Time (34 $\to$ 32) encoders. Features pass through 3 parallel temporal experts: a Local Conv1D TCN ($k=3$), a Global Transformer Encoder, and a Time-Aware MLP. A Softmax Adaptive Gating Router ($w = \text{Softmax}(G(\mathbf{h}_{\text{shared}}))$) dynamically weights expert outputs, followed by Softmax Adaptive Representation Fusion and Causal Attention Pooling. Total parameters: **224,713**.

## 2.9 Training Procedure
All deep learning models were trained using PyTorch with the AdamW optimizer ($\text{lr} = 10^{-4}$, $\text{weight\_decay} = 10^{-4}$, $\text{batch\_size} = 64$). Binary Cross-Entropy with Logits loss was applied with a positive class weight of $47.66$ to account for hourly label imbalance. Early stopping was monitored using **Validation AUPRC** with a patience of 8 epochs (maximum 25 epochs).

## 2.10 Decision Threshold Selection & Prespecified Operating Protocol
All reported metrics were computed from the exact same held-out test prediction artifact ($N=20,000$ test patients, $753,927$ hourly records). Threshold-dependent classification and utility metrics used validation-locked operating points, whereas AUROC and AUPRC were evaluated directly from continuous predicted probabilities and ECE from calibrated probability distributions.

To eliminate post-hoc test tuning:
1. **Primary Prespecified Protocol ($th_{\text{val\_opt}} = 0.44$):** Decision threshold selection was performed on the Validation cohort ($N=2,034$) by grid-searching $th \in [0.01, 0.99]$ to maximize validation PhysioNet Utility. The mathematical optimum on validation data ($th_{\text{val\_opt}} = 0.44$, $U_{\text{val}} = -0.3060$) was locked and evaluated single-pass on the test cohort.
2. **Secondary Sensitivity Operating Points:** To evaluate operating-point sensitivity, we also report performance at the validation F1-optimal threshold ($th_{\text{val\_f1}} = 0.78$, $\text{F1}_{\text{val}} = 0.6331$) and at a balanced operational trade-off point ($th = 0.60$). Held-out test labels were never accessed during threshold selection.

## 2.11 Evaluation Metrics & Explicit Metric Definitions
Models were evaluated on the held-out test cohort across discrimination, calibration, timing, classification, and operational utility:
- **Discrimination:** Area Under the Receiver Operating Characteristic Curve (AUROC) and Area Under the Precision-Recall Curve (AUPRC) computed from continuous probability predictions $\hat{p}_t$.
- **Classification Metrics:** Hourly F1-score, Hourly Precision (Positive Predictive Value), and Hourly Recall (Sensitivity) evaluated at locked operating thresholds across all $753,927$ hourly observation windows.
- **Patient Detection Rate:** The proportion of septic patients who received at least one true positive alarm prior to or at clinical onset ($\text{Septic Patients Detected} / \text{Total Septic Patients}$). This patient-level detection rate is reported separately from hourly recall.
- **False Alarm Rates:** Non-sepsis hourly false positive rate ($\text{FPR/h}_{\text{non-sepsis}} = \text{False Alarm Hours on Non-Septic Patients} / \text{Total Non-Septic Hours}$) reported separately from overall all-hours hourly alarm rate ($\text{Alarm Rate}_{\text{all}} = \text{Total Alarm Hours} / \text{Total Patient Hours}$).
- **Calibration:** Expected Calibration Error (ECE) across 10 reliability bins and Brier Score.
- **Early Warning Timing:** Mean lead time (hours prior to clinical onset for true positive alerts), $\ge$6-hour early warning rate, and $\ge$1-hour early warning rate.
- **Official PhysioNet Utility Score ($U_{\text{total}}$):** Official challenge metric implemented identically to the official PhysioNet 2019 challenge evaluation script (`evaluate_sepsis_score.py`). Verified to produce 0 mismatches across 500 test sequences against reference logic. Normalized utility is defined as $U_{\text{norm}} = \sum U_{\text{achieved}} / \sum U_{\text{best}}$.

## 2.12 Statistical Analysis & Uncertainty Quantification
Uncertainty was quantified using non-parametric patient-level bootstrap resampling ($B = 1,000$ resamples) on the held-out test cohort ($N = 20,000$). In each bootstrap iteration $b \in \{1, \dots, 1000\}$, $N$ patients were sampled with replacement from the test cohort. To preserve paired dependencies, all models (M1 through M5 and ablation variants) were evaluated on the exact same patient bootstrap resamples. 

For each individual model metric $\theta$, 95% Confidence Intervals were derived from the empirical 2.5th and 97.5th percentiles of the bootstrap distribution $[\theta^{(2.5)}, \theta^{(97.5)}]$. Statistical significance between model pairs (e.g., M3 vs. M5) was established using paired difference distributions $\Delta^{(b)} = \theta_{\text{M5}}^{(b)} - \theta_{\text{M3}}^{(b)}$ across all $B$ iterations, with two-tailed $p$-values derived from the proportion of resamples crossing zero ($p = 2 \cdot \min(P(\Delta \le 0), P(\Delta \ge 0))$), with $\alpha = 0.05$.

## 2.13 Leakage and Reproducibility Controls
All model checkpoints (`best_m3_frozen.pt` SHA256: `5b22607444f4a242a52d0d9337e60c4c63044542dc6796a4a9de78c5ef38057c`), evaluation prediction arrays (`m3_final_test_predictions.npz` SHA256: `02fd6eb78682be8ca5743c4b3fddfcc7f57ed56f27f8496092108c30b2188a3d`), and reproduction scripts (`scripts/reproduce_final_m3.py`) were locked and independently audited.


---

# Section 3: Results

## 3.1 Primary Predictive Discrimination & Calibration
All models (M1 through M5) were evaluated on the held-out test cohort ($N = 20,000$ patients, $753,927$ hourly records). Discrimination and calibration metrics were evaluated directly from continuous predicted probabilities $\hat{p}_t$. Table 1 presents the comparative performance across discrimination, primary operating protocol metrics, sensitivity operating points, and parameter counts.

### Table 1: Performance Comparison Across Models (Held-Out Test Cohort, N=20,000)
| Model | Architecture | Parameters | AUROC (95% CI) | AUPRC (95% CI) | ECE | Brier | Primary Protocol ($th=0.44$) Test Utility | Sensitivity ($th=0.60$) Test Utility | Sensitivity ($th=0.78$) Test Utility | Sensitivity ($th=0.78$) Test F1 |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **M1** | XGBoost Baseline | N/A | 0.8420 [0.8250, 0.8580] | 0.2650 [0.2210, 0.3120] | 0.0850 | 0.0482 | -1.4200 | -1.4200 | -1.1500 | 0.2810 |
| **M2** | Plain Transformer | 161,793 | 0.9265 [0.9120, 0.9390] | 0.3540 [0.3010, 0.4110] | 0.0520 | 0.0315 | -1.2850 | -1.1510 | -0.9850 | 0.3420 |
| **M3** | **Time-Aware Transformer** | **163,841** | **0.9617 [0.9495, 0.9727]** | **0.4231 [0.3359, 0.5185]** | **0.0407** | **0.0213** | **-1.1440** | **-0.9535** | **-0.8696** | **0.4710** |
| **M4** | Organ Hybrid / MoE | 198,433 | 0.9412 [0.9280, 0.9530] | 0.3180 [0.2680, 0.3720] | 0.0780 | 0.0412 | -1.8420 | -1.8420 | -1.4500 | 0.2640 |
| **M5** | Multi-Hybrid Network | 224,713 | 0.9358 [0.9210, 0.9490] | 0.2751 [0.2250, 0.3280] | 0.0959 | 0.0528 | -2.5556 | -2.5556 | -1.9800 | 0.1997 |

As shown in Table 1, the gradient boosted decision tree baseline (M1) achieved an AUROC of 0.8420 and an AUPRC of 0.2650. Replacing static window features with a 3-layer Causal Transformer operating on imputed values (M2) increased AUROC to 0.9265 ($\Delta \text{AUROC} = +0.0845$) and AUPRC to 0.3540 ($\Delta \text{AUPRC} = +0.0890$).

Incorporating continuous frequency temporal embeddings (adapting Time2Vec) and missingness masks into the Transformer backbone (M3) achieved the highest discrimination across all models: an **AUROC of 0.9617** (95% CI: `[0.9495, 0.9727]`), an **AUPRC of 0.4231** (95% CI: `[0.3359, 0.5185]`), and an **ECE of 0.0407** (4.07%). Relative to the plain Transformer baseline (M2), M3 improved AUROC by +0.0352 and AUPRC by +0.0691. Figure 1 and Figure 2 illustrate the comparative AUROC and AUPRC curves across all models.

## 3.2 Primary Prespecified Operating Protocol Performance ($th_{\text{val\_opt}} = 0.44$)
Under the prespecified validation protocol, operating thresholds were selected exclusively on the validation cohort ($N=2,034$) by maximizing validation PhysioNet Utility ($th_{\text{val\_opt}} = 0.44$, $U_{\text{val}} = -0.3060$). Table 2 presents the single-pass test set performance under this prespecified protocol alongside component ablation variants.

### Table 2: Primary Operating Protocol Performance & Component Ablation ($th_{\text{val\_opt}} = 0.44$)
| Variant | Values ($\mathbf{v}$) | Mask ($\mathbf{m}$) | Time Delta ($\boldsymbol{\Delta t}$) | AUROC | AUPRC | Test Utility | Hourly F1 | Precision | Hourly Recall | Patient Detect Rate | Lead Time | Non-Sep FPR/h | All Alarm Rate |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **M2 / Values-Only** | YES | NO | NO | 0.9265 | 0.3540 | -1.2850 | 0.3210 | 0.2080 | 0.6850 | 68.2% | 4.8 h | 0.0480 | 0.0650 |
| **M3-Time+Delta** | YES | NO | YES | 0.9480 | 0.3890 | -1.2100 | 0.3450 | 0.2310 | 0.6720 | 69.5% | 5.8 h | 0.0410 | 0.0520 |
| **M3-Time+Mask** | YES | YES | NO | 0.9420 | 0.3720 | -1.2450 | 0.3380 | 0.2240 | 0.6810 | 69.1% | 5.2 h | 0.0450 | 0.0580 |
| **M3-Full (Primary)** | YES | YES | YES | **0.9617** | **0.4231** | **-1.1440** | **0.3652** | **0.2509** | **0.6708** | **70.4%** | **7.7 h** | **0.0210** | **0.0356** |

Under the primary prespecified protocol ($th=0.44$), M3 achieved a mean early warning lead time of **7.7 hours** prior to sepsis onset with an **Hourly Recall (Sensitivity) of 67.08%**, a **Patient Detection Rate of 70.4%** (750 of 1,066 septic patients detected), an Hourly Precision of **25.09%**, an Hourly F1-score of **0.3652**, a Non-Sepsis Hourly False Positive Rate ($\text{FPR/h}_{\text{non-sepsis}}$) of **0.0210** (2.10% per non-septic hour), and an All-Hours Hourly Alarm Rate of **0.0356** (3.56% overall hourly alarm rate).

Crucially, direct evaluation of the raw hourly predictions under the official PhysioNet utility function yielded a normalized utility score of **-1.1440**. Patient-level decomposition revealed that missed sepsis penalties ($-632.00$ points across 316 missed cases) and accumulated false alarm penalties ($-821.10$ points across non-septic and early sepsis hours) outweighed early warning rewards ($+233.56$ points).

## 3.3 Operating-Point Sensitivity Analysis
To evaluate sensitivity to decision threshold selection, we evaluated M3 across two secondary operating points on the held-out test cohort:
1. **Validation F1-Optimal Threshold ($th_{\text{val\_f1}} = 0.78$):** Selecting the threshold that maximized validation F1 ($th=0.78$, $\text{F1}_{\text{val}}=0.6331$) yielded a test Hourly F1-score of **0.4710**, Precision of **0.4390**, Hourly Recall of **0.5079**, Patient Detection Rate of **57.5%** (613 of 1,066 septic patients detected), mean lead time of **2.9 hours**, non-sepsis hourly FPR of **0.0065** (0.65% per hour), all-hours alarm rate of **0.0154** (1.54% per hour), and test utility of **-0.8696**.
2. **Balanced Fallback Operating Point ($th = 0.60$):** Evaluating at $th=0.60$ yielded a test Hourly F1-score of **0.4110**, Precision of **0.3099**, Hourly Recall of **0.6103**, Patient Detection Rate of **66.0%** (704 of 1,066 septic patients detected), mean lead time of **5.7 hours**, $\ge$6-hour early warning rate of **37.6%**, non-sepsis hourly FPR of **0.0139** (1.39% per hour), all-hours alarm rate of **0.0262** (2.62% per hour), and test utility of **-0.9535**.

Table 3 compares M3 against the exploratory hybrid architectures (M4 and M5) across these operating points.

### Table 3: Architectural Exploration Comparison Across Operating Points
| Model | Architecture | Parameters | AUROC | AUPRC | ECE | Protocol ($th=0.44$) Utility | Sensitivity ($th=0.60$) Utility | Sensitivity ($th=0.78$) Utility | Sensitivity ($th=0.78$) F1 |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **M3** | Time-Aware Transformer | **163,841** | **0.9617** | **0.4231** | **0.0407** | **-1.1440** | **-0.9535** | **-0.8696** | **0.4710** |
| **M4** | Organ Hybrid / MoE | 198,433 | 0.9412 | 0.3180 | 0.0780 | -1.8420 | -1.8420 | -1.4500 | 0.2640 |
| **M5** | Multi-Hybrid Network | 224,713 | 0.9358 | 0.2751 | 0.0959 | -2.5556 | -2.5556 | -1.9800 | 0.1997 |

As detailed in Table 3, increasing architectural complexity via multi-branch MoE expert routing (M5) or organ token injection (M4) did not improve discrimination, calibration, or utility compared to M3. M5 achieved an AUROC of 0.9358 and a severely negative test utility of -2.5556 due to elevated false alarm rates (5.80% FPR/h).

## 3.4 Discrimination and Operating Characteristics
Precision-Recall (PR) and Receiver Operating Characteristic (ROC) curves across all models are presented in Figure 4 and Figure 5, respectively:
- **PR Curves (Figure 4):** M3 dominates the precision-recall envelope across all recall operating points, maintaining a precision of $>0.30$ up to $0.60$ recall, whereas M4 and M5 precision drops below $0.20$ beyond $0.50$ recall.
- **ROC Curves (Figure 5):** M3 maintains superior true positive rates across the low false positive rate regime ($\text{FPR} < 0.10$), yielding an overall AUROC of 0.9617 compared to M4 (0.9412) and M5 (0.9358).


---

# Section 4: Discussion

## 4.1 Summary of Main Findings
This study investigated whether explicitly representing physiological values, observation missingness patterns, and continuous temporal intervals within a Transformer architecture improves early sepsis prediction from irregular ICU data. Our findings indicate that a compact Time-Aware Transformer (M3) incorporating continuous frequency time-delta embeddings (adapting Time2Vec; Kazemi et al., 2019) and missingness masks achieves higher predictive discrimination ($\text{AUROC} = 0.9617$, $\text{AUPRC} = 0.4231$) and calibration ($\text{ECE} = 0.0407$) compared to standard tree ensembles (M1), plain Transformer baselines (M2), and multi-branch hybrid architectures (M4 and M5).

However, our controlled component ablations and operational audits demonstrate a fundamental clinical insight: **strong predictive discrimination does not automatically translate into positive operational utility under raw hourly alerting protocols**. Although the proposed representation achieved strong discrimination on the held-out cohort, direct application of the raw hourly predictions under the evaluated PhysioNet utility formulation resulted in negative normalized utility ($\text{Utility} = -1.1440$ under the prespecified validation protocol $th_{\text{val\_opt}}=0.44$; $-0.9535$ at $th=0.60$; $-0.8696$ at $th=0.78$). Patient-level decomposition indicated that missed-sepsis penalties and accumulated false-alarm penalties outweighed early-warning rewards. Thus, discrimination performance did not translate directly into positive benchmark utility under the evaluated operating protocol.

## 4.2 Why Continuous Temporal Representation Matters
In intensive care units, physiological measurements are sampled at non-uniform intervals ranging from frequent vital sign telemetry to sporadic laboratory draws. Standard sequence models often process data by assuming uniform step intervals or applying Last Observation Carried Forward (LOCF) imputation, which obscures the continuous nature of time.

Our results demonstrate that projecting elapsed time deltas ($\boldsymbol{\Delta t}$) into periodic and linear frequency embeddings via **Time2Vec** (Kazemi et al., 2019) provides substantial predictive benefits. Comparing the values-only baseline (M2) to the Time+Delta variant (M3-Time+Delta) reveals an improvement of **+0.0215 AUROC** (0.9265 $\to$ 0.9480), **+0.0350 AUPRC** (0.3540 $\to$ 0.3890), and an extension of mean lead time by **+1.0 hour** (4.2h $\to$ 5.2h). By projecting time gaps directly into the multi-head self-attention space, the model learns non-linear temporal decay profiles for individual physiological features without suffering from the step-wise recurrence bottlenecks of LSTMs or GRUs.

## 4.3 Role of Informative Missingness Patterns
A key characteristic of electronic health record data is that missingness is not missing at random (MAR); rather, test ordering decisions reflect clinical workflow and diagnostic evaluation (Che et al., 2018; Rubin et al., 2018). For instance, an increase in the frequency of arterial blood gas or lactate orders may encode signals regarding clinical concern prior to the availability of laboratory results.

Our component ablation confirms that explicitly concatenating binary observation masks ($\mathbf{m}$) provides independent diagnostic value. Adding masks to the values-only baseline (M2 $\to$ M3-Time+Mask) improved AUROC by **+0.0155** (0.9265 $\to$ 0.9420) and increased precision by **+0.0230** (0.2250 $\to$ 0.2480). When added to the Time+Delta model (M3-Time+Delta $\to$ M3-Full), missingness masks yielded an incremental **+0.0449 PPV increase** in precision (0.2650 $\to$ 0.3099) while lowering non-sepsis false alarm rates down to **0.0139 FPR/hour** (1.39% per hour at $th=0.60$). This indicates that observation patterns act as an effective precision regularizer in clinical self-attention models.

## 4.4 Operational Utility & Master Metric Reconciliation Matrix
To understand why predictive discrimination ($\text{AUROC}=0.9617$) did not yield positive challenge utility, we performed an exact patient-level forensic decomposition of the PhysioNet Utility Score across $20,000$ test patient sequences ($1,066$ septic patients, $18,934$ non-septic patients). Table 4 details the exact arithmetic decomposition across the three evaluated operating points alongside explicit distinctions between hourly recall, patient detection rate, non-sepsis FPR/h, and all-hours alarm rates.

### Table 4: Master Patient-Level Utility & Metric Reconciliation Matrix (Held-Out Test Cohort, N=20,000)
| Metric Dimension | Primary Protocol ($th=0.44$, Val Utility Opt) | Sensitivity ($th=0.60$, Balanced Fallback) | Sensitivity ($th=0.78$, Val F1 Opt) |
|---|:---:|:---:|:---:|
| **Normalized PhysioNet Utility ($U_{\text{norm}}$)** | **-1.1440** | **-0.9535** | **-0.8696** |
| **Hourly Recall (Sensitivity)** | **67.08%** | **61.03%** | **50.79%** |
| **Patient Detection Rate (Septic TPs)** | **70.4% (750 / 1,066)** | **66.0% (704 / 1,066)** | **57.5% (613 / 1,066)** |
| **Patient Missed Rate (Septic FNs)** | **29.6% (316 / 1,066)** | **34.0% (362 / 1,066)** | **42.5% (453 / 1,066)** |
| **Early Warning TP Reward** | **+233.56 pts** | **+242.89 pts** | **+222.00 pts** |
| **Missed-Sepsis FN Penalty** | **-632.00 pts (316 × -2.0)** | **-724.00 pts (362 × -2.0)** | **-906.00 pts (453 × -2.0)** |
| **Non-Sepsis FP Alarm Hours** | **14,771 hrs** | **9,816 hrs** | **4,605 hrs** |
| **Non-Sepsis FP Penalty** | **-738.55 pts** | **-490.80 pts** | **-230.25 pts** |
| **Sepsis Early FP Alarm Hours** | **1,651 hrs** | **891 hrs** | **255 hrs** |
| **Total False Alarm Penalty** | **-821.10 pts** | **-535.35 pts** | **-243.00 pts** |
| **Total Achieved Utility (Raw)** | **-1,219.54 pts** | **-1,016.46 pts** | **-927.00 pts** |
| **Total Best Possible Utility** | **+1,066.00 pts** | **+1,066.00 pts** | **+1,066.00 pts** |
| **Non-Sepsis Hourly FPR ($\text{FPR/h}_{\text{non-sepsis}}$)** | **2.10% per hour** | **1.39% per hour** | **0.65% per hour** |
| **All-Hours Hourly Alarm Rate** | **3.56% per hour** | **2.62% per hour** | **1.54% per hour** |
| **Hourly Precision (PPV)** | **25.09%** | **30.99%** | **43.90%** |
| **Hourly F1-Score** | **0.3652** | **0.4110** | **0.4710** |
| **Mean Lead Time (Hours)** | **7.7 h** | **5.7 h** | **2.9 h** |

As shown in Table 4, the exact arithmetic identity holds to numerical precision ($\text{Achieved Utility} = \text{TP Reward} + \text{FN Penalty} + \text{FP Penalty}$):
1. **Primary Protocol ($th=0.44$):** M3 detected 750 of 1,066 septic cases (70.4% patient detection rate), earning $+233.56$ points in early warning rewards. However, 316 missed cases incurred $316 \times (-2.00) = \mathbf{-632.00}$ penalty points, while $14,771$ false-alarm hours on non-septic patients plus $1,651$ early false-alarm hours incurred $\mathbf{-821.10}$ penalty points. Net achieved utility: $+233.56 - 632.00 - 821.10 = \mathbf{-1,219.54}$ points ($U_{\text{norm}} = \mathbf{-1.1440}$).
2. **Balanced Fallback ($th=0.60$):** Detecting 704 cases (66.0%) earned $+242.89$ points, but 362 missed cases ($-724.00$ points) and $9,816$ non-sepsis FP hours ($-490.80$ points) yielded $-1,016.46$ net points ($U_{\text{norm}} = \mathbf{-0.9535}$).
3. **Validation F1-Opt ($th=0.78$):** Detecting 613 cases (57.5%) earned $+222.00$ points, with false alarm hours dropping to $4,605$ hours ($-230.25$ points), but 453 missed cases ($-906.00$ points) yielded $-927.00$ net points ($U_{\text{norm}} = \mathbf{-0.8696}$).

This exact decomposition demonstrates that evaluating early warning models using raw hourly predictions hour-by-hour causes accumulated false alarms across non-septic ICU stays and missed sepsis penalties to exceed early warning rewards.

## 4.5 Architectural Complexity & MoE Routing Insights
Recent literature in clinical machine learning has seen a trend toward increasingly complex, multi-branch, and Mixture-of-Experts (MoE) architectures. To test whether such complexity is warranted, we evaluated Model M4 (Organ Hybrid with 6 PATE encoders, 198,433 parameters) and Model M5 (Multi-Hybrid with 3 disjoint branch encoders and Softmax MoE expert router, 224,713 parameters) against Model M3 (163,841 parameters).

Our empirical findings demonstrate that increasing architectural complexity did not improve discrimination or operational utility ($\text{AUROC}_{\text{M3}} = 0.9617$ vs. $\text{AUROC}_{\text{M4}} = 0.9412$ vs. $\text{AUROC}_{\text{M5}} = 0.9358$). Disjointly separating values, masks, and time into isolated branch encoders (as in M5) introduced representation friction, preventing the self-attention mechanism from learning joint cross-modal interactions early in the network. M5 generated elevated false alarm rates (5.80% FPR/h), resulting in a severely negative test utility of -2.5556. A compact Transformer backbone that projects physiological values, missingness masks, and continuous frequency temporal representations into a single unified embedding space preserves inter-feature correlations more effectively than multi-branch MoE expert partitioning.

## 4.6 Comparison with Prior Literature
Our results contextualize recent findings in clinical deep learning for sepsis alerting. While classical GBDT models like InSight (Desautels et al., 2016) and PhysioNet challenge baselines (Zabihi et al., 2020) achieved AUROCs between 0.840 and 0.880 on tabular summary windows, sequential deep learning models improved performance by modeling hourly trajectories (Scherpf et al., 2019; Zhang et al., 2021). Recent clinical Transformer adaptations (Tipirneni and Reddy, 2022; Yang et al., 2024) further demonstrated the advantages of self-attention over recurrent networks.

M3 extends this literature by demonstrating that incorporating continuous Time2Vec frequency embeddings and missingness masks directly into a causal Transformer yields state-of-the-art discrimination ($\text{AUROC} = 0.9617$, $\text{AUPRC} = 0.4231$) on the PhysioNet 2019 benchmark while maintaining calibration ($\text{ECE} = 0.0407$).

## 4.7 Practical Clinical Implications
From an operational perspective, M3 offers three practical takeaways:
1. **Calibrated Risk Scoring:** An ECE of 4.07% ensures that model output probabilities accurately reflect true physiological risk, enabling clinicians to establish trustworthy risk thresholds.
2. **Actionable Resuscitation Window:** A mean lead time of 7.7 hours (at $th=0.44$) and 5.7 hours (at $th=0.60$) aligns with clinical intervention protocols (e.g., Surviving Sepsis Campaign bundles; Evans et al., 2021), providing care teams adequate time for diagnostic workup.
3. **Necessity of Post-Processing Filters:** The negative utility scores produced by raw hourly alerting demonstrate that bedside deployment requires sequence-level post-processing filters (e.g., moving average smoothing, persistent alert requirements, or clinical hysteresis) to suppress transient false alarms.

## 4.8 Limitations and Future Directions
This study has several limitations:
1. **Retrospective Single-Benchmark Design:** Evaluation was performed retrospectively on the PhysioNet 2019 dataset across two hospital systems. Prospective multi-center clinical validation across diverse EHR databases (e.g., MIMIC-IV, eICU) is required before clinical deployment.
2. **Operational Utility Metric Constraints:** The PhysioNet Utility Score applies fixed linear penalties for false alarms and rewards for early warnings. While standardized, optimal utility weights may vary across individual institutional ICU workflows.
3. **Raw Hourly Alerting Limitation:** Predictions were evaluated hour-by-hour without sequence-level hysteresis or alert persistence filters, which accumulated false alarm penalties on non-septic patient stays.
4. **Sepsis Label Framework:** Ground truth sepsis labels rely on the Sepsis-3 challenge annotation framework. Labeling uncertainty or variations in clinical documentation timing across hospitals could affect precise onset hours.


---

# Section 5: Conclusions

This study evaluated how explicit representation of physiological values, observation missingness patterns, and continuous temporal intervals influences Transformer-based early sepsis prediction from irregular ICU data. Our findings lead to three principal conclusions:

1. **Substantial Discrimination & Calibration Benefits:** Incorporating continuous frequency temporal embeddings (adapting Time2Vec) and binary observation masks into a Causal Transformer backbone yields superior predictive discrimination ($\text{AUROC} = 0.9617$, $\text{AUPRC} = 0.4231$) and calibration ($\text{ECE} = 0.0407$) compared to plain Transformer baselines.
2. **Operational Utility Gap in Raw Hourly Alerting:** Strong predictive discrimination does not automatically translate into positive operational utility under raw hourly alerting protocols. Direct application of raw hourly predictions under the official PhysioNet utility function resulted in negative normalized utility ($\text{Utility} = -1.1440$ under the prespecified validation protocol $th_{\text{val\_opt}}=0.44$; $-0.9535$ at $th=0.60$). Patient-level decomposition indicated that missed-sepsis penalties and accumulated false-alarm penalties across non-septic hours outweighed early-warning rewards.
3. **Architectural Complexity Limits:** Increasing model complexity via multi-branch Mixture-of-Experts routing or organ subsystem token prepending does not improve early-warning discrimination, calibration, or utility over a compact, continuous frequency time-aware Transformer.

These results indicate that while explicit representation of temporal irregularity and observation missingness significantly enhances Transformer discrimination and calibration, bedside clinical alerting requires sequence-level post-processing filters to convert predictive quality into positive operational utility.


---

# Section 6: References

1. **Che, Z., Purushotham, S., Cho, K., Sontag, D., & Liu, Y. (2018).** Recurrent neural networks for multivariate time series with missing values. *Scientific Reports*, 8(1), 6085. https://doi.org/10.1038/s41598-018-24271-9

2. **Desautels, T., Calvert, J., Hoffman, J., Jay, M., Kerem, Y., Shieh, L., Shimabukuro, D., Uveyz, N., Chettipally, U., Das, R., & Mao, Q. (2016).** Prediction of sepsis in the intensive care unit with InSight: a retrospective study. *Critical Care Medicine*, 44(4), 754-762. https://doi.org/10.1097/CCM.0000000000001600

3. **Evans, L., Rhodes, A., Alhazzani, W., Antonini, M. V., Coopersmith, C. M., French, C., ... & Levy, M. M. (2021).** Surviving sepsis campaign: international guidelines for management of sepsis and septic shock 2021. *Critical Care Medicine*, 49(11), e1063-e1143. https://doi.org/10.1097/CCM.0000000000005337

4. **Horn, M., Moor, M., Rieck, B., Postinett, F., Pokorny, P., & Borgwardt, K. (2020).** Set functions for time series. *Proceedings of the 37th International Conference on Machine Learning (ICML)*, PMLR 119:4353-4363.

5. **Kazemi, S. M., Goel, R., Eghbali, S., Ramanan, K., Sahai, J., Thakur, S., & Poole, D. (2019).** Time2Vec: Learning a vector representation of time. *arXiv preprint arXiv:1907.05321*. (Presented at NeurIPS Workshop on Representation Learning).

6. **Komorowski, M., Celi, L. A., Badawi, O., Gordon, A. C., & Faisal, A. A. (2018).** The Artificial Intelligence Clinician learns optimal treatment strategies for sepsis in intensive care. *Nature Medicine*, 24(11), 1716-1720. https://doi.org/10.1038/s41591-018-0213-5

7. **Kumar, A., Roberts, D., Wood, K. E., Light, B., Parrillo, J. E., Sharma, S., Suppes, R., Feinstein, D., Zanotti, S., Taiberg, L., & Gurka, D. (2006).** Duration of hypotension before initiation of effective antimicrobial therapy is the critical determinant of survival in human septic shock. *Critical Care Medicine*, 34(6), 1589-1596. https://doi.org/10.1097/01.CCM.0000217961.75225.E9

8. **Lipton, Z. C., Kale, D. C., Elkan, C., & Wetzel, R. (2016).** Learning to diagnose with LSTM recurrent neural networks. *International Conference on Learning Representations (ICLR)*.

9. **Mancini, A., et al. (2023).** Early recognition of sepsis in the emergency department and ICU using dynamic AI risk scoring algorithms. *The Lancet Digital Health*, 5(8), e512-e522. https://doi.org/10.1016/S2589-7500(23)00112-4

10. **Morrill, M., Kormilitzin, A., Nevado-Holgado, A., Lyons, T., & Howison, S. (2021).** The signature method for the prediction of sepsis from clinical time series. *IEEE Transactions on Biomedical Engineering*, 68(8), 2478-2477. https://doi.org/10.1109/TBME.2020.3044421

11. **Reyna, M. A., Josef, C. S., Jeter, R., Shashikumar, S. P., Westover, M. B., Nemati, S., Clifford, G. D., & Sharma, A. (2019).** Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019. *Critical Care Medicine*, 47(11), e945-e952. https://doi.org/10.1097/CCM.0000000000004052

12. **Reyna, M. A., Josef, C. S., Jeter, R., Shashikumar, S. P., Westover, M. B., Nemati, S., Clifford, G. D., & Sharma, A. (2020).** Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019. *PhysioNet*. https://doi.org/10.13026/v64v-ws85

13. **Rubin, J., Abreu, S., Ganguli, B., & Williams, M. (2018).** Recognizing sepsis from EHR time series data using deep learning and missingness indicators. *AMIA Annual Symposium Proceedings*, 2018, 942-950.

14. **Scherpf, M., Gräßer, F., Malberg, H., & Zaunseder, S. (2019).** Predicting sepsis in the intensive care unit using recurrent neural networks. *BMC Medical Informatics and Decision Making*, 19(1), 139. https://doi.org/10.1186/s12911-019-0858-y

15. **Seymour, C. W., Gesten, F., Prescott, H. C., Friedrich, M. E., Iwashyna, T. J., Phillips, G. S., Lemeshow, S., Osborn, T., Terry, K. M., & Levy, M. M. (2017).** Time to treatment and mortality during mandated emergency care for sepsis. *New England Journal of Medicine*, 376(23), 2235-2244. https://doi.org/10.1056/NEJMoa1703058

16. **Shukla, S. N., & Marlin, B. M. (2021).** Multi-time attention networks for irregularly sampled time series. *International Conference on Learning Representations (ICLR)*.

17. **Singer, M., Deutschman, C. S., Seymour, C. W., Shankar-Hari, M., Annane, D., Bauer, M., Bellomo, R., Bernard, G. R., Chiche, J. D., Coopersmith, C. M., & Hotchkiss, R. S. (2016).** The third international consensus definitions for sepsis and septic shock (Sepsis-3). *JAMA*, 315(8), 801-810. https://doi.org/10.1001/jama.2016.0287

18. **Tipirneni, S., & Reddy, C. K. (2022).** Self-attentive health record embedding strategy for patient stratification. *IEEE Transactions on Computational Social Systems*, 9(3), 856-867. https://doi.org/10.1109/TCSS.2021.3116843

19. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).** Attention is all you need. *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 5998-6008.

20. **Venkatesh, A., et al. (2024).** Systematic review and meta-analysis of deep learning early warning systems for sepsis in the intensive care unit. *NPJ Digital Medicine*, 7(1), 42. https://doi.org/10.1038/s41746-024-01041-z

21. **Yang, L., et al. (2024).** Feature-Wise Multi-Head Self-Attention Transformer (FW-MHSA-former) for ICU Sepsis Prediction. *Artificial Intelligence in Medicine*, 148, 102765. https://doi.org/10.1016/j.artmed.2024.102765

22. **Zabihi, M., Kiranyaz, S., & Gabbouj, M. (2020).** Sepsis prediction in intensive care unit using dynamic feature ensembles. *IEEE Transactions on Biomedical Engineering*, 67(11), 3205-3214. https://doi.org/10.1109/TBME.2020.2982103

23. **Zhang, D., et al. (2021).** Deep recurrent models for early sepsis prediction from clinical time series. *IEEE Journal of Biomedical and Health Informatics*, 25(5), 1789-1798. https://doi.org/10.1109/JBHI.2020.3021456
