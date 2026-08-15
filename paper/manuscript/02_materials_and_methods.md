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
