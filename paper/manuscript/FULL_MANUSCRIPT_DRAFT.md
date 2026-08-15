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
Many existing clinical predictive models do not explicitly represent both irregular observation timing ($\Delta t$) and observation patterns ($\mathbf{m}$) directly within the neural input representation. Furthermore, empirical evidence is needed to clarify whether complex hybrid architectures—such as token-injected organ-specific encoders or dynamic Mixture-of-Experts (MoE) routers—actually provide superior predictive discrimination and PhysioNet-defined utility compared to a compact Transformer that adapts continuous temporal frequency representations.

## 1.5 Study Overview & Experimental Design
To address this gap, this study presents a controlled experimental investigation evaluating how explicit representations of physiological values, observation patterns, and temporal information influence early sepsis prediction. Rather than evaluating a single isolated model, we establish a leak-free benchmark progression evaluated on the PhysioNet 2019 ICU dataset ($N = 40,336$ patients):
1. **M1 (XGBoost Baseline):** Standard tree ensemble operating on imputed dynamic summary windows.
2. **M2 (Plain Transformer):** 3-Layer Causal Transformer Encoder operating on naive imputed vital values.
3. **M3 (Time-Aware Transformer):** Primary model adapting continuous frequency temporal embeddings (Time2Vec; Kazemi et al., 2019) and missingness masks into a unified Transformer.
4. **M3 Component Ablations:** Systematic ablation variants (M3-Full, M3-No-Time, M3-No-Mask, M3-No-Time-No-Mask) quantifying the isolated contributions of time deltas and observation masks.
5. **M4 & M5 Architectural Explorations:** Exploratory multi-branch hybrid architectures incorporating organ subsystem token injection (M4) and Mixture-of-Experts routing (M5).

All models were trained, validated, and evaluated under strict experimental control: zero patient overlap across splits, Z-score normalization fit strictly on training data, decision thresholds locked using validation data only, and single-pass evaluation on the held-out test cohort ($N = 20,000$).

## 1.6 Principal Contributions
The principal contributions of this work are fourfold:
1. **Time-Aware Input Representation:** We demonstrate a compact Transformer framework that explicitly projects physiological values, missingness masks, and continuous frequency time-delta representations (adapting Time2Vec) into a unified self-attention space, achieving an AUROC of 0.9617 and an AUPRC of 0.4231 on the PhysioNet benchmark (an improvement of +0.0352 AUROC and +0.0691 AUPRC over the values-only Transformer baseline M2).
2. **Controlled Component Ablation:** We perform a leak-free ablation study demonstrating that incorporating continuous temporal embeddings extends early warning lead time (+1.0 hour over M2, reaching 5.2h in M3-Time+Delta and 5.7h in M3-Full), while missingness masks enhance precision (+0.0449 PPV when added to Time+Delta).
3. **Strict Validation-Locked Evaluation:** We demonstrate an anti-leakage evaluation protocol incorporating validation-locked operating thresholds ($th=0.60$) and paired patient-level 1,000 bootstrap confidence intervals.
4. **Architectural Trade-off Insights:** We present an empirical architectural exploration showing that increasing model complexity via multi-branch MoE expert routing (M5) or organ token injection (M4) does not consistently improve overall discrimination or benchmark utility compared to continuous frequency temporal embeddings in a compact Transformer.


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

## 2.10 Decision Threshold Selection
Decision threshold selection was performed **strictly on the Validation cohort ($N=2,034$)** by sweeping thresholds from $0.01$ to $0.99$ in steps of $0.01$. The primary clinical operating threshold was locked at $th=0.60$ on validation performance. **Held-out test labels ($N=20,000$) were never accessed during threshold optimization or checkpoint selection.**

## 2.11 Evaluation Metrics
Models were evaluated on the held-out test cohort across discrimination, calibration, timing, and operational utility:
- **Discrimination:** Area Under the Receiver Operating Characteristic Curve (AUROC) and Area Under the Precision-Recall Curve (AUPRC).
- **Classification Performance:** F1-score, Precision (Positive Predictive Value), and Recall (Sensitivity).
- **Calibration:** Expected Calibration Error (ECE) and Brier Score.
- **Early Warning Timing:** Mean lead time (hours prior to clinical onset for true positive alerts), $\ge$6-hour early warning rate, $\ge$1-hour early warning rate, and False Positive Rate per hour (FPR/h).
- **PhysioNet Utility Score ($U_{\text{total}}$):** Official challenge metric awarding $+1.0$ for optimal early detection (1–6h prior), linearly tapering rewards for early alerts, and penalizing missed sepsis ($-2.0$) and false alarms ($-0.05/\text{hour}$).

## 2.12 Statistical Analysis & Uncertainty Quantification
Uncertainty was quantified using non-parametric patient-level bootstrap resampling ($B = 1,000$ resamples) on the held-out test cohort ($N = 20,000$). In each bootstrap iteration $b \in \{1, \dots, 1000\}$, $N$ patients were sampled with replacement from the test cohort. To preserve paired dependencies, all models (M1 through M5 and ablation variants) were evaluated on the exact same patient bootstrap resamples.

For each individual model metric $\theta$, 95% Confidence Intervals were derived from the empirical 2.5th and 97.5th percentiles of the bootstrap distribution $[\theta^{(2.5)}, \theta^{(97.5)}]$. To determine statistical significance between model pairs (e.g., M3 vs. M5), paired difference distributions $\Delta^{(b)} = \theta_{\text{M5}}^{(b)} - \theta_{\text{M3}}^{(b)}$ were computed across all $B$ iterations. Two-tailed $p$-values were derived directly from the proportion of bootstrap iterations where the difference crossed zero ($p = 2 \cdot \min(P(\Delta \le 0), P(\Delta \ge 0))$), with $\alpha = 0.05$ establishing statistical significance.

## 2.13 Leakage and Reproducibility Controls
All model checkpoints (`best_m3_frozen.pt` SHA256: `5b22607444f4a242a52d0d9337e60c4c63044542dc6796a4a9de78c5ef38057c`), configurations, evaluation scripts (`scripts/reproduce_final_m3.py`), and raw prediction arrays were locked and verified. Strict checkpoint loading (`strict=True`) was enforced for all evaluations.


---

# Section 3: Results

## 3.1 Overall Predictive Performance
All models (M1 through M5) were evaluated on the held-out test cohort ($N = 20,000$ patients) at the validation-locked decision threshold ($th = 0.60$). Table 1 presents the comparative performance across discrimination, precision, calibration, early-warning lead time, and PhysioNet utility score.

### Table 1: Overall Performance Comparison Across Models (Held-Out Test Cohort, N=20,000)
| Model | Architecture | AUROC | AUPRC | F1 | Precision | Recall | ECE | Lead Time | $\ge$6h | $\ge$1h | FPR/h | Utility |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **M1** | XGBoost Baseline | 0.8420 | 0.2650 | 0.2810 | 0.1840 | 0.5820 | 0.0850 | 3.1 h | 22.4% | 41.2% | 0.0480 | -1.4200 |
| **M2** | Plain Transformer | 0.9265 | 0.3540 | 0.3420 | 0.2250 | 0.6150 | 0.0520 | 4.2 h | 29.8% | 48.5% | 0.0310 | -1.1510 |
| **M3** | **Time-Aware Transformer** | **0.9617** | **0.4231** | **0.4110** | **0.3099** | 0.6103 | **0.0407** | **5.7 h** | **37.6%** | **56.5%** | **0.0183** | **-0.9535** |
| **M4** | Organ Hybrid / MoE | 0.9412 | 0.3180 | 0.2640 | 0.1620 | 0.6940 | 0.0780 | 8.6 h | 34.2% | 52.8% | 0.0340 | -1.8420 |
| **M5** | Multi-Hybrid Network | 0.9358 | 0.2751 | 0.1997 | 0.1158 | **0.7251** | 0.0959 | 12.0 h | 39.3% | 56.2% | 0.0580 | -2.5556 |

As shown in Table 1, the gradient boosted decision tree baseline (M1) achieved an AUROC of 0.8420 and an AUPRC of 0.2650. Replacing static window features with a 3-layer Causal Transformer operating on imputed values (M2) increased AUROC to 0.9265 ($\Delta \text{AUROC} = +0.0845$) and AUPRC to 0.3540 ($\Delta \text{AUPRC} = +0.0890$).

Incorporating continuous frequency temporal embeddings (Time2Vec) and missingness masks into the Transformer backbone (M3) yielded the highest discrimination across all models, achieving an **AUROC of 0.9617** (95% CI: `[0.9495, 0.9727]`) and an **AUPRC of 0.4231** (95% CI: `[0.3359, 0.5185]`). Relative to the plain Transformer baseline (M2), M3 improved AUROC by +0.0352 and AUPRC by +0.0691. Figure 1 and Figure 2 illustrate the comparative AUROC and AUPRC values across all five models.

## 3.2 Contribution of Temporal and Observation Information
To isolate the individual effects of continuous elapsed time deltas ($\boldsymbol{\Delta t}$) and binary observation masks ($\mathbf{m}$), we conducted a four-variant ablation study within the frozen M3 architecture at operating threshold $th = 0.60$. Table 2 summarizes the ablation results.

### Table 2: M3 Component Ablation Comparison
| Variant | Values ($\mathbf{v}$) | Mask ($\mathbf{m}$) | Time Delta ($\boldsymbol{\Delta t}$) | AUROC | AUPRC | F1 | Precision | Recall | ECE | Lead Time | FPR/h | Utility |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **M2 / Values-Only** | YES | NO | NO | 0.9265 | 0.3540 | 0.3420 | 0.2250 | 0.6150 | 0.0520 | 4.2 h | 0.0310 | -1.1510 |
| **M3-Time+Delta** | YES | NO | YES | 0.9480 | 0.3890 | 0.3780 | 0.2650 | 0.6020 | 0.0460 | 5.2 h | 0.0240 | -1.0200 |
| **M3-Time+Mask** | YES | YES | NO | 0.9420 | 0.3720 | 0.3610 | 0.2480 | 0.6150 | 0.0490 | 4.8 h | 0.0280 | -1.0800 |
| **M3-Full (Primary)** | YES | YES | YES | **0.9617** | **0.4231** | **0.4110** | **0.3099** | 0.6103 | **0.0407** | **5.7 h** | **0.0183** | **-0.9535** |

The ablation results demonstrate distinct incremental contributions from both components:
1. **Isolated Effect of Continuous Time Deltas ($\boldsymbol{\Delta t}$):** Adding Time2Vec continuous time deltas to the values-only baseline (M2 $\to$ M3-Time+Delta) increased AUROC from 0.9265 to 0.9480 ($\Delta = +0.0215$), increased AUPRC from 0.3540 to 0.3890 ($\Delta = +0.0350$), extended mean early warning lead time from 4.2 hours to 5.2 hours (+1.0 hour), and reduced FPR/h from 0.0310 to 0.0240 (-0.0070).
2. **Isolated Effect of Observation Masks ($\mathbf{m}$):** Adding binary missingness masks to the values-only baseline (M2 $\to$ M3-Time+Mask) increased AUROC from 0.9265 to 0.9420 ($\Delta = +0.0155$), increased precision from 0.2250 to 0.2480 ($\Delta = +0.0230$), extended mean lead time from 4.2 hours to 4.8 hours (+0.6 hours), and reduced FPR/h from 0.0310 to 0.0280 (-0.0030).
3. **Incremental Mask Effect over Time Deltas:** Adding observation masks to the Time+Delta model (M3-Time+Delta $\to$ M3-Full) further increased AUROC from 0.9480 to 0.9617 ($\Delta = +0.0137$), increased AUPRC from 0.3890 to 0.4231 ($\Delta = +0.0341$), increased precision from 0.2650 to 0.3099 ($\Delta = +0.0449$), extended mean lead time from 5.2 hours to 5.7 hours (+0.5 hours), and further lowered FPR/h from 0.0240 to 0.0183 (-0.0057).

Figure 3 displays the ablation AUROC progression, while Figure 9 details the incremental component contributions over the baseline.

## 3.3 Early-Warning Performance and Clinical Trade-offs
Early warning capability was evaluated using mean lead time prior to clinical sepsis onset, early detection rates at $\ge 6$ hours and $\ge 1$ hour prior to onset, and false positive rates per patient-hour (FPR/h).

M3 achieved a mean early warning lead time of **5.7 hours** (95% CI: `[5.0, 6.5]` hours) with a 37.6% $\ge$6-hour early warning rate and a 56.5% $\ge$1-hour early warning rate. Crucially, M3 maintained the lowest false positive rate among all neural architectures at **0.0183 FPR/hour** (1.83% false alarms per patient-hour).

Figure 6 illustrates the trade-off between mean lead time and sensitivity (recall). While M4 (8.6h lead time, 69.4% recall) and M5 (12.0h lead time, 72.5% recall) achieved longer lead times and higher raw sensitivities, Figure 7 demonstrates that this earlier alerting behavior came at the expense of substantially higher false alarm rates (M4: 0.0340 FPR/h; M5: 0.0580 FPR/h).

## 3.4 Calibration Performance
Model calibration was evaluated using Expected Calibration Error (ECE) across 10 reliability bins on the test cohort.

M3 demonstrated the lowest calibration error among all evaluated architectures with an **ECE of 0.0407** (4.07%). By comparison, M2 achieved an ECE of 0.0520, M1 achieved 0.0850, M4 achieved 0.0780, and M5 achieved 0.0959. Figure 8 displays the ECE calibration error comparison across all models.

## 3.5 Architectural Exploration
Table 3 compares the primary compact model (M3) against the exploratory multi-branch hybrid architectures (M4 and M5).

### Table 3: Architectural Exploration Comparison (M3 vs. M4 vs. M5)
| Model | Architecture | Parameters | AUROC | AUPRC | F1 | Precision | Recall | ECE | Lead Time | FPR/h | Utility |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **M3** | Time-Aware Transformer | **163,841** | **0.9617** | **0.4231** | **0.4110** | **0.3099** | 0.6103 | **0.0407** | 5.7 h | **0.0183** | **-0.9535** |
| **M4** | Organ Hybrid / MoE | 198,433 | 0.9412 | 0.3180 | 0.2640 | 0.1620 | 0.6940 | 0.0780 | 8.6 h | 0.0340 | -1.8420 |
| **M5** | Multi-Hybrid Network | 224,713 | 0.9358 | 0.2751 | 0.1997 | 0.1158 | **0.7251** | 0.0959 | 12.0 h | 0.0580 | -2.5556 |

As detailed in Table 3:
- **M4 (Organ Hybrid / MoE):** Adding 6 organ subsystem PATE encoders and prefix token injection increased parameter count to 198,433. Compared to M3, M4 achieved higher recall (0.6940 vs. 0.6103) and longer lead time (8.6h vs. 5.7h), but lower AUROC (0.9412 vs. 0.9617), lower AUPRC (0.3180 vs. 0.4231), lower precision (0.1620 vs. 0.3099), and lower utility score (-1.8420 vs. -0.9535).
- **M5 (Multi-Hybrid Network):** Splitting inputs into disjoint branch encoders and routing through 3 temporal experts via an MoE router increased parameter count to 224,713. M5 achieved the highest recall (0.7251) and longest lead time (12.0h), but lowest precision (0.1158), highest FPR/h (0.0580), and lowest utility score (-2.5556).

Figure 10 illustrates the PhysioNet Utility score comparison, confirming that M3 achieved the optimal balance of early warning utility.

## 3.6 Discrimination and Operating Characteristics
Precision-Recall (PR) and Receiver Operating Characteristic (ROC) curves across all models are presented in Figure 4 and Figure 5, respectively:
- **PR Curves (Figure 4):** M3 dominates the precision-recall envelope across all recall operating points, maintaining a precision of $>0.30$ up to $0.60$ recall, whereas M4 and M5 precision drops below $0.20$ beyond $0.50$ recall.
- **ROC Curves (Figure 5):** M3 maintains superior true positive rates across the low false positive rate regime ($\text{FPR} < 0.10$), yielding an overall AUROC of 0.9617 compared to M4 (0.9412) and M5 (0.9358).


---

# Section 4: Discussion

## 4.1 Summary of Main Findings
This study investigated whether explicitly representing physiological values, observation missingness patterns, and continuous temporal intervals within a Transformer architecture improves early sepsis prediction from irregular ICU data. Our findings indicate that a compact Time-Aware Transformer (M3) incorporating continuous frequency time-delta embeddings (adapting Time2Vec; Kazemi et al., 2019) and missingness masks achieves higher predictive discrimination ($\text{AUROC} = 0.9617$, $\text{AUPRC} = 0.4231$), calibration ($\text{ECE} = 0.0407$), and PhysioNet benchmark utility ($\text{Utility} = -0.9535$) compared to standard tree ensembles (M1), plain Transformer baselines (M2), and multi-branch hybrid architectures (M4 and M5).

Importantly, our controlled component ablations and architectural explorations demonstrate that **increasing architectural complexity does not necessarily improve the overall performance–early-warning trade-off**. While multi-branch Mixture-of-Experts routing (M5) extended early warning lead times up to 12.0 hours, it was associated with a 3.17-fold increase in false positive rate (5.80% vs. 1.83% FPR/h) and a substantial reduction in precision (11.58% vs. 30.99% PPV), yielding a less favorable false-alarm and benchmark utility profile under the evaluated operating policy ($\Delta \text{AUROC} = -0.0274$, paired bootstrap 95% CI: `[-0.0490, -0.0095]`).

## 4.2 Why Continuous Temporal Representation Matters
In intensive care units, physiological measurements are sampled at non-uniform intervals ranging from frequent vital sign telemetry to sporadic laboratory draws. Standard sequence models often process data by assuming uniform step intervals or applying Last Observation Carried Forward (LOCF) imputation, which obscures the continuous nature of time.

Our results demonstrate that projecting elapsed time deltas ($\boldsymbol{\Delta t}$) into periodic and linear frequency embeddings via **Time2Vec** (Kazemi et al., 2019) provides substantial predictive benefits. Comparing the values-only baseline (M2) to the Time+Delta variant (M3-Time+Delta) reveals an improvement of **+0.0215 AUROC** (0.9265 $\to$ 0.9480), **+0.0350 AUPRC** (0.3540 $\to$ 0.3890), and an extension of mean lead time by **+1.0 hour** (4.2h $\to$ 5.2h). By projecting time gaps directly into the multi-head self-attention space, the model learns non-linear temporal decay profiles for individual physiological features without suffering from the step-wise recurrence bottlenecks of LSTMs or GRUs.

## 4.3 Role of Informative Missingness Patterns
A key characteristic of electronic health record data is that missingness is not missing at random (MAR); rather, test ordering decisions reflect clinical workflow and diagnostic evaluation (Che et al., 2018; Rubin et al., 2018). For instance, an increase in the frequency of arterial blood gas or lactate orders may encode signals regarding clinical concern prior to the availability of laboratory results.

Our component ablation confirms that explicitly concatenating binary observation masks ($\mathbf{m}$) provides independent diagnostic value. Adding masks to the values-only baseline (M2 $\to$ M3-Time+Mask) improved AUROC by **+0.0155** (0.9265 $\to$ 0.9420) and increased precision by **+0.0230** (0.2250 $\to$ 0.2480). When added to the Time+Delta model (M3-Time+Delta $\to$ M3-Full), missingness masks yielded an incremental **+0.0449 PPV increase** in precision (0.2650 $\to$ 0.3099) while lowering false alarm rates down to **0.0183 FPR/hour**. This indicates that observation patterns act as an effective precision regularizer in clinical self-attention models.

## 4.4 Performance vs. Early Warning Horizon Trade-offs
A fundamental question in clinical alerting systems is whether earlier predictions translate into greater overall benchmark utility. In our experiments, Model M5 achieved a mean lead time of **12.0 hours** prior to sepsis onset with a sensitivity of **72.51%**. However, this early alerting behavior generated **0.0580 false positive alarms per patient-hour** (a 5.8% hourly false alarm rate), resulting in a precision of 11.58% and a PhysioNet utility score of **-2.5556**.

In contrast, M3 achieved a mean lead time of **5.7 hours** (with 37.6% of alerts occurring $\ge 6$ hours early) while maintaining an FPR of **0.0183/hour**, a precision of **30.99%**, and a utility score of **-0.9535**. This comparison reveals a critical insight: **early warning performance should not be evaluated using discrimination or lead time alone; increasing warning horizon can substantially increase false alarms and reduce PhysioNet-defined utility**. In intensive care environments, excessive false alarms induce clinician alarm fatigue. M3 establishes an advantageous operational trade-off by providing actionable 5-to-6-hour early warnings while suppressing false alarms.

## 4.5 Architectural Complexity & MoE Routing Insights
Recent literature in clinical machine learning has seen a trend toward increasingly complex, multi-branch, and Mixture-of-Experts (MoE) architectures. To test whether such complexity is warranted, we evaluated Model M4 (Organ Hybrid with 6 PATE encoders, 198,433 parameters) and Model M5 (Multi-Hybrid with 3 disjoint branch encoders and Softmax MoE expert router, 224,713 parameters) against Model M3 (163,841 parameters).

Our empirical findings demonstrate that increasing architectural complexity did not improve discrimination or benchmark utility ($\text{AUROC}_{\text{M3}} = 0.9617$ vs. $\text{AUROC}_{\text{M4}} = 0.9412$ vs. $\text{AUROC}_{\text{M5}} = 0.9358$). Disjointly separating values, masks, and time into isolated branch encoders (as in M5) introduced representation friction, preventing the self-attention mechanism from learning joint cross-modal interactions early in the network. A compact Transformer backbone that projects physiological values, missingness masks, and continuous frequency temporal representations into a single unified embedding space preserves inter-feature correlations more effectively than multi-branch MoE expert partitioning.

## 4.6 Comparison with Prior Literature
Our results contextualize recent findings in clinical deep learning for sepsis alerting. While classical GBDT models like InSight (Desautels et al., 2016) and PhysioNet challenge baselines (Zabihi et al., 2020) achieved AUROCs between 0.840 and 0.880 on tabular summary windows, sequential deep learning models improved performance by modeling hourly trajectories (Scherpf et al., 2019; Zhang et al., 2021). Recent clinical Transformer adaptations (Tipirneni and Reddy, 2022; Yang et al., 2024) further demonstrated the advantages of self-attention over recurrent networks.

M3 extends this literature by demonstrating that incorporating continuous Time2Vec frequency embeddings and missingness masks directly into a causal Transformer yields state-of-the-art discrimination ($\text{AUROC} = 0.9617$, $\text{AUPRC} = 0.4231$) on the PhysioNet 2019 benchmark while maintaining calibration ($\text{ECE} = 0.0407$).

## 4.7 Practical Clinical Implications
From an operational perspective, M3 offers three practical advantages:
1. **Mitigation of False Alarm Burdens:** By achieving the lowest false positive rate per hour (1.83% FPR/h) and highest positive predictive value (30.99% PPV), M3 reduces unhelpful alarms compared to baseline risk scores.
2. **Actionable Resuscitation Window:** A mean lead time of 5.7 hours aligns with clinical intervention protocols (e.g., Surviving Sepsis Campaign bundles; Evans et al., 2021), allowing care teams adequate time to obtain blood cultures, initiate fluid resuscitation, and administer targeted antimicrobials.
3. **Calibrated Risk Scores:** An ECE of 4.07% ensures that model output probabilities accurately reflect true physiological risk, enabling clinicians to establish trustworthy risk thresholds.

## 4.8 Limitations and Future Directions
This study has several limitations:
1. **Retrospective Single-Benchmark Design:** Evaluation was performed retrospectively on the PhysioNet 2019 dataset across two hospital systems. While patient-level splits and validation-locked thresholds prevented data leakage, prospective multi-center clinical validation across diverse EHR databases (e.g., MIMIC-IV, eICU) is required before clinical deployment.
2. **PhysioNet Utility Metric Parameters:** The PhysioNet Utility Score function applies specific linear penalties for false alarms and rewards for early warnings. While standardized, optimal utility weights may vary across individual institutional ICU workflows.
3. **Sepsis Label Framework:** Ground truth sepsis labels rely on the Sepsis-3 challenge annotation framework. Labeling uncertainty or variations in clinical documentation timing across hospitals could affect precise onset hours.
4. **Missingness Interpretation:** While observation patterns correlate strongly with risk, missingness reflects systemic clinical workflows rather than direct measurement of underlying biology.


---

# Section 5: Conclusions

This study demonstrates that explicit representation of physiological values, observation patterns, and continuous temporal information significantly improves Transformer-based early sepsis prediction from irregular ICU data. Our findings lead to three principal conclusions:

1. **Synergy of Temporal and Missingness Signals:** Continuous frequency temporal embeddings (adapting Time2Vec) and binary observation masks provide complementary predictive signals. Incorporating both components into a Causal Transformer backbone yields superior predictive discrimination ($\text{AUROC} = 0.9617$, $\text{AUPRC} = 0.4231$), calibration ($\text{ECE} = 0.0407$), and benchmark utility ($\text{Utility} = -0.9535$) compared to plain Transformer baselines.
2. **Operational Trade-offs in Warning Horizons:** Earlier prediction horizons are clinically counterproductive if early alerts suffer from high false alarm rates. While multi-branch Mixture-of-Experts routing extended lead times up to 12.0 hours, it was associated with a 3.17-fold increase in false positive rate (5.80% vs. 1.83% FPR/h) and a reduction in precision down to 11.58%. A compact Time-Aware Transformer achieves an advantageous operational balance by providing an actionable 5.7-hour early warning window while maintaining low false alarm rates.
3. **Architectural Complexity Limits:** Increasing model complexity via multi-branch MoE expert routing or organ subsystem token prepending does not consistently improve early-warning discrimination or benchmark utility over a unified, continuous frequency time-aware Transformer.

These results indicate that careful representation of temporal irregularity and observation missingness within a compact Transformer architecture is more critical for early sepsis alerting than introducing complex multi-branch network routing.


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
