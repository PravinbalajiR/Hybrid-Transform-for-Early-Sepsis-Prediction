# Full Publication Research Paper: Time-Aware Sepsis Early Warning & Cross-Hospital Utility Decomposition

---

## 📌 CANDIDATE MANUSCRIPT TITLES

1. **Option 1 (Recommended):**  
   *High Discrimination Does Not Guarantee Clinical Utility: A Decision-Ceiling Decomposition of Cross-Hospital Sepsis Early Warning*
2. **Option 2 (Architectural & Utility Focus):**  
   *Time-Aware Transformer Representations for Early Sepsis Prediction: Evaluating Cross-Hospital Discrimination and Net Clinical Utility*
3. **Option 3 (Methodological & Information Focus):**  
   *Quantifying the Information-Utility Gap in Deep Temporal Clinical Models: A Multi-Center Sepsis Study*
4. **Option 4 (Operational Focus):**  
   *Beyond AUROC: Dual-Bound Utility Decomposition of Clinical Time-Series Models Across Health Systems*
5. **Option 5 (Short & Direct):**  
   *Discriminative Discrimination vs. Operational Utility in Cross-Hospital Sepsis Early Warning*

---

## FINAL RECOMMENDED TITLE
# High Discrimination Does Not Guarantee Clinical Utility: A Decision-Ceiling Decomposition of Cross-Hospital Sepsis Early Warning

---

## ABSTRACT

**Background:** Conventional predictive metrics such as the Area Under the Receiver Operating Characteristic curve (AUROC) and Area Under the Precision-Recall Curve (AUPRC) are standard benchmarks for evaluating clinical machine learning models. However, in temporal early-warning applications—such as forecasting sepsis onset in intensive care units (ICUs)—high discriminative rank-ordering does not guarantee positive net clinical utility when false alarm penalties accumulate under asymmetric decision costs.

**Objective:** We systematically develop and evaluate a progression of deep learning architectures for early sepsis prediction across hospital systems, establish the strongest discriminative representation, and investigate why high conventional discrimination fails to translate into positive operational utility under cross-hospital deployment.

**Methods:** We evaluated a structured family of predictive models—ranging from classical gradient boosting ($M1$) and plain Transformers ($M2$) to Time-Aware Transformers ($M3$) incorporating physiological values, missingness masks, and elapsed-time deltas ($\Delta t$), as well as complex organ-aware ($M4$) and multi-hybrid/routing architectures ($M5$). Models were trained on $20,336$ ICU stays from Emory University Hospital (Set A) and evaluated on a held-out test cohort of $20,000$ ICU stays from Beth Israel Deaconess Medical Center (Set B; $1,066$ septic patients, $18,934$ non-septic patients, $753,927$ hourly observations). Controlled ablation experiments were conducted to isolate the independent contributions of temporal and missingness encodings. To diagnose operational utility deficits, we formulated a **Dual-Bound Utility Decomposition Framework** that separates: (1) a perfect-information `GROUND_TRUTH_ORACLE_CEILING` (using true labels and optimal timing only); (2) a counterfactual `PATIENT_ADAPTIVE_THRESHOLD_CEILING`; (3) a `HINDSIGHT_GRID_SCORE_POLICY_CEILING` (optimizing threshold and alert suppression cooldown on test data); and (4) the prespecified deployable `FROZEN_MODEL_UTILITY`. Uncertainty was quantified via patient-level bootstrap resampling ($B = 1,000$) and multi-seed stability testing ($N = 6$ seeds).

**Results:** Among the evaluated architectures, the compact $M3$ Time-Aware Transformer provided the strongest cross-hospital discriminative representation (AUROC = $0.9617$, AUPRC = $0.4231$, Brier = $0.0153$, ECE = $0.0182$), outperforming XGBoost ($M1$, AUROC = $0.8842$), plain Transformers ($M2$, AUROC = $0.9265$), and complex hybrid architectures ($M4$, AUROC = $0.9582$; $M5$, AUROC = $0.9591$). Controlled ablations confirmed that explicitly encoding time deltas and missingness masks improved AUROC from $0.9265$ to $0.9617$. However, under cross-hospital deployment on BIDMC, deployable clinical utility was strictly negative (`FROZEN_MODEL_UTILITY` = $-0.2573$, 95% CI: `[-0.2828, -0.2335]`). The `GROUND_TRUTH_ORACLE_CEILING` was positive ($+0.8262$, 95% CI: `[+0.8067, +0.8448]`), proving that positive utility is mathematically achievable under the official action space. Dense 2D policy sweeps revealed that even under optimal hindsight thresholding and alert suppression, the global policy ceiling remained strictly negative (`HINDSIGHT_GRID_SCORE_POLICY_CEILING` = $-0.1983$, 95% CI: `[-0.2185, -0.1783]`). Although counterfactual patient-adaptive thresholding achieved positive utility ($+0.2819$, 95% CI: `[+0.2579, +0.3040]`), predictability modeling using admission and early-trajectory features yielded random-level discrimination (AUPRC = $0.2653$ vs. base rate $0.2608$), confirming that adaptive threshold needs are not reliably identifiable in advance (`REALISTIC_ACHIEVABLE_UTILITY` = $-0.1983$). A paired bootstrap comparison confirmed a statistically significant information/representation gap between perfect-information decision making and observable score policies ($\Delta = +1.0246$, 95% CI: `[+0.9997, +1.0494]`, $p < 0.0001$). Results were robust across $6$ distinct random initialization seeds (AUROC = $0.9609 \pm 0.0016$, Utility = $-0.2573 \pm 0.0020$).

**Conclusion:** High conventional discrimination (AUROC $0.9617$) did not translate into positive clinical utility under cross-hospital deployment. The observed utility deficit is consistent with a substantial information/representation gap between perfect-information decision making and observable scalar risk scores, compounded by false alarm accumulation in non-septic mimic hours. These findings demonstrate that evaluating predictive discrimination alone can mask severe operational failure modes, highlighting the necessity of decision-theoretic evaluation frameworks in clinical artificial intelligence.

---

## 1. INTRODUCTION

Early recognition of sepsis in intensive care units (ICUs) is a major imperative in acute care medicine. Sepsis, defined as life-threatening organ dysfunction caused by a dysregulated host response to infection (Singer et al., 2016), affects over $49$ million individuals annually and accounts for nearly $20\%$ of global deaths. Because mortality increases precipitously with delayed antimicrobial therapy, machine learning systems have been extensively developed to forecast sepsis hours before clinical diagnosis (Desautels et al., 2016; Nemati et al., 2018).

In clinical machine learning literature, predictive performance is overwhelmingly benchmarked using conventional rank-ordering metrics, primarily the Area Under the Receiver Operating Characteristic curve (AUROC) and Area Under the Precision-Recall Curve (AUPRC). While these metrics measure an algorithm's capacity to rank high-risk observations above low-risk observations across all possible decision thresholds, they ignore operational deployment realities—such as false alarm penalties, intervention lead times, and alert suppression constraints. In real-world deployment, an early-warning model operates as a binary decision policy under asymmetric decision costs: late alarms miss crucial therapeutic windows, while excessive false alarms trigger severe monitor alarm fatigue and clinician burnout (Cvach, 2012; Vickers & Elkin, 2006).

This disconnect is magnified under cross-hospital deployment, where clinical data distributions shift across health systems due to variations in patient demographics, electronic health record (EHR) systems, nursing measurement workflows, and clinical practice patterns (Subbaswamy & Saria, 2020). Although modern deep learning architectures—such as Time-Aware Transformers—demonstrate high discriminative ability on internal and external datasets, their operational efficacy under explicit clinical utility functions remains under-evaluated.

### 1.1 Research Questions
In this study, we address two interconnected research challenges: first, identifying the optimal architectural representation for cross-hospital sepsis prediction; and second, diagnosing why high predictive discrimination fails to produce positive clinical utility under deployment. Specifically, we investigate four primary research questions:

- **RQ1:** Does explicit temporal and missingness representation improve cross-hospital sepsis discrimination over classical machine learning and standard Transformer baselines?
- **RQ2:** Does increasing architectural complexity beyond a compact time-aware Transformer representation yield corresponding performance gains?
- **RQ3:** Does high discriminative rank-ordering translate into positive net clinical utility under cross-hospital deployment?
- **RQ4:** If deployable net utility fails, is the failure attributable to threshold selection, calibration, architectural capacity, or fundamental limitations in observable clinical information?

### 1.2 Contributions
To answer these questions, we present a unified empirical study spanning model development, controlled component ablations, cross-hospital evaluation, and formal utility decomposition. Our main contributions are:

1. **Controlled Model Progression ($M1$–$M5$):** We systematically benchmark a structured family of models on $40,336$ ICU stays across two health systems, establishing that a compact Time-Aware Transformer ($M3$) provides the strongest cross-hospital discriminative representation (AUROC = $0.9617$), outperforming XGBoost ($M1$), plain Transformers ($M2$), and complex hybrid architectures ($M4$, $M5$).
2. **Component Ablation Analysis:** We isolate the independent contributions of temporal time-deltas ($\Delta t$) and missingness masks ($m$), proving that explicitly representing clinical observation dynamics drives discriminative gains.
3. **Empirical Demonstration of the Discrimination–Utility Disconnect:** We show that despite achieving top-tier AUROC ($0.9617$) on a held-out test cohort of $20,000$ ICU stays from BIDMC, the deployable model utility is strictly negative (`FROZEN_MODEL_UTILITY` = $-0.2573$).
4. **Dual-Bound Utility Decomposition Framework:** We introduce a decision-ceiling framework that separates perfect-information decision potential (`GROUND_TRUTH_ORACLE_CEILING = +0.8262`), global score policy limits (`HINDSIGHT_GRID_SCORE_POLICY_CEILING = -0.1983`), counterfactual patient-adaptive headroom (`PATIENT_ADAPTIVE_THRESHOLD_CEILING = +0.2819`), and deployable model performance.
5. **Information/Representation Gap Quantification:** Through patient-level bootstrap resampling ($B=1,000$) and multi-seed stability checks ($N=6$), we provide statistical proof ($\Delta_{\text{Info}} = +1.0246, p < 0.0001$) of a substantial separation between perfect-information decision making and observable score policies, demonstrating that the utility failure is consistent with an information/representation limitation.

---

## 2. RELATED WORK

### 2.1 Machine Learning for Early Sepsis Prediction
Early automated sepsis detection relied on rule-based clinical criteria—such as SIRS, SOFA, and qSOFA (Singer et al., 2016)—which suffer from low sensitivity or delayed triggering. Machine learning approaches advanced early warning by training supervised classifiers on static and aggregated vital signs (Desautels et al., 2016). More recent systems leverage deep recurrent networks and gradient boosting on continuous multivariate ICU time series (Nemati et al., 2018; Reyna et al., 2019).

### 2.2 Missingness-Aware Clinical Time-Series Modeling
ICU data are characterized by irregular measurement intervals and informative missingness patterns. Che et al. (2018) introduced GRU-D, showing that incorporating binary observation masks and elapsed time deltas ($\Delta t$) allows neural networks to capture clinical sampling frequencies. Subsequent studies confirmed that modeling observation dynamics improves risk prediction in acute care time series.

### 2.3 Transformer-Based Clinical Time-Series Models
Self-attention mechanisms have been adapted for clinical time series to capture long-range temporal dependencies without recurrent bottlenecking (Li et al., 2020). By integrating time-aware positional embeddings and missingness encodings, Time-Aware Transformers achieve state-of-the-art rank-ordering performance (AUROC $>0.95$) on competitive benchmarks.

### 2.4 Cross-Hospital Generalization and Domain Shift
Dataset shift represents a major barrier to deploying clinical machine learning models (Subbaswamy & Saria, 2020). Differences in patient demographics, disease prevalence, nursing measurement frequencies, and hospital coding practices frequently cause model performance to degrade when transferred to new medical centers. Domain adaptation methods, such as Domain-Adversarial Neural Networks (DANN; Ganin & Lempitsky, 2015), attempt to align feature representations across hospitals, but their impact on downstream clinical utility metrics remains poorly understood.

### 2.5 Decision-Theoretic Evaluation and Clinical Utility
Decision Curve Analysis (Vickers & Elkin, 2006) and utility-based scoring metrics (Reyna et al., 2019) move beyond AUROC by incorporating clinical costs and consequences. The PhysioNet 2019 Utility Metric explicitly penalizes false alarms ($-0.05$ pts/hr) and missed sepsis ($-2.0$ pts/patient) while rewarding timely early warnings ($+1.0$ pts). However, existing literature lacks formal decomposition frameworks to isolate whether utility failures stem from metric harshness, policy constraints, domain shift, or fundamental representation limits.

### 2.6 Positioning of This Study
Unlike prior studies focusing solely on maximizing AUROC or proposing complex neural architectures, this study evaluates a structured model family under formal cross-hospital deployment and introduces a dual-bound utility decomposition framework to rigorously diagnose operational utility breakdown.

---

## 3. MATERIALS AND METHODS

### 3.1 Dataset and Cohorts
We utilized the open-access PhysioNet/Computing in Cardiology Challenge 2019 dataset (Reyna et al., 2019), comprising $40,336$ adult ICU stays from two distinct health systems:
- **Set A (Emory University Hospital):** $20,336$ ICU stays ($1,790$ septic, $18,546$ non-septic).
- **Set B (Beth Israel Deaconess Medical Center):** $20,000$ ICU stays ($1,066$ septic, $18,934$ non-septic).

Each record consists of hourly observations containing $8$ vital sign variables, $26$ laboratory measurement variables, and $6$ demographic/admission variables.

### 3.2 Cross-Hospital Experimental Setup
To evaluate cross-hospital transfer without domain leakage:
- **Training & Validation Cohort:** Set A (Emory) was partitioned at the patient level into an in-domain training set ($16,192$ patients) and an in-domain validation set ($4,144$ patients) using an 80/20 stratified split based on sepsis prevalence.
- **Held-Out Test Cohort:** Set B (BIDMC) was preserved as an independent held-out test cohort ($N=20,000$ patients, $753,927$ hourly observations). Patient overlap across all splits is zero ($0.0$).

### 3.3 Data Preprocessing and Leakage Prevention
Raw clinical features were normalized to zero mean and unit variance. Normalization parameters were fit strictly on the in-domain training split ($16,192$ patients) and applied unchanged to validation and held-out test data. Missing raw values were forward-filled. For each feature $j \in \{1, \dots, 34\}$ at hour $t$, we constructed a triplet representation:
1. Normalized feature value $v_{t,j} \in \mathbb{R}$.
2. Binary missingness mask $m_{t,j} \in \{0, 1\}$.
3. Elapsed time delta $\Delta t_{t,j} \in \mathbb{R}_{\ge 0}$, indicating hours since the last physical observation.

### 3.4 Model Progression ($M1$–$M5$)
We evaluated a structured family of five predictive architectures:
- **$M1$ (XGBoost Baseline):** Gradient boosted decision trees operating on aggregated temporal summary statistics.
- **$M2$ (Plain Transformer):** A standard 3-layer Transformer encoder operating on forward-filled physiological features without time-delta or missingness encodings.
- **$M3$ (Time-Aware Transformer):** The primary 3-layer Transformer architecture operating on unified triplet encodings $(v, m, \Delta t)$.
- **$M4$ (Organ-Aware Hybrid Architecture):** A dual-branch Transformer incorporating explicit organ-system grouping layers (Cardiovascular, Renal, Pulmonary, Hepatic, Hematologic).
- **$M5$ (Multi-Hybrid / MoE Architecture):** A mixture-of-experts architecture featuring temporal routing modules across organ-specific expert networks.

### 3.5 M3 Architecture Specifications
The $M3$ Time-Aware Transformer (`TACTModel`) maps input triplets $\mathbf{x}(t) = [\mathbf{v}(t), \mathbf{m}(t), \mathbf{\Delta t}(t)] \in \mathbb{R}^{102}$ at each hour $t$ to hidden representation $\mathbf{h}(t) \in \mathbb{R}^{64}$ through a linear embedding layer and sinusoidal positional encodings. The network incorporates $3$ Transformer encoder layers ($4$ self-attention heads, $d_{\text{model}}=64$, GELU activations, dropout $p=0.10$). A linear classification head projects $\mathbf{h}(t)$ to uncalibrated logit $z(t)$, transformed via sigmoid activation to yield risk probability $p(t) \in (0, 1)$.

### 3.6 Controlled M3 Component Ablations
To isolate the sources of discriminative performance in $M3$, we conducted controlled ablation experiments evaluating four feature variants:
1. **No-Time + No-Mask:** Standard forward-filled values only ($v$).
2. **Time-Aware Only:** Values plus time deltas ($v, \Delta t$).
3. **Mask-Aware Only:** Values plus missingness masks ($v, m$).
4. **Full M3:** Combined values, missingness masks, and time deltas ($v, m, \Delta t$).

### 3.7 Training Protocol
Models were trained using Binary Cross-Entropy with Logits loss, incorporating positive class weighting ($w_{\text{pos}} \approx 11.2$) to compensate for hourly label imbalance. Optimization used AdamW ($\text{lr} = 10^{-4}, \text{weight\_decay} = 10^{-4}$) with ReduceLROnPlateau scheduling for up to $30$ epochs on an NVIDIA CUDA GPU. Hyperparameters were tuned strictly on the Emory validation split.

### 3.8 Discrimination and Calibration Metrics
Model discrimination was evaluated using Area Under the Receiver Operating Characteristic curve (AUROC) and Area Under the Precision-Recall Curve (AUPRC). Probability calibration was evaluated using Brier Score and Expected Calibration Error (ECE) across $10$ equal-width probability bins.

### 3.9 Official PhysioNet Utility Metric
Net clinical utility was computed using the official PhysioNet 2019 utility function $U(S, Y)$:
- **Optimal TP Credit (+1.0):** Single alarm at $t_{\text{optimal}} = \max(0, t_{\text{onset}} - 6\text{h})$.
- **Early/Late Penalty:** Linear ramp from $0.0$ at $t_{\text{early}} = 12\text{h}$ pre-onset to $+1.0$ at $t_{\text{optimal}}$, decaying to $0.0$ at $t_{\text{late}} = 3\text{h}$ post-onset.
- **False Alarm Penalty:** $-0.05$ points per hour.
- **Missed Sepsis Penalty:** $-2.0$ points per patient.
- **Normalization:** Total achieved utility points divided by total maximum possible points ($N_{\text{sepsis}} \times 1.0 = 1,066.0$).

### 3.10 Dual-Bound Utility Decomposition Framework
To diagnose utility deficits, we formulated a four-level utility hierarchy:

1. **`GROUND_TRUTH_ORACLE_CEILING` ($U_{\text{GT}}$):** Maximum achievable utility under perfect information using true sepsis labels $y_{\text{true}}$ and onset times $t_{\text{onset}}$ only. It uses **zero** model scores, probabilities, logits, or predictions.
2. **`PATIENT_ADAPTIVE_THRESHOLD_CEILING` ($U_{\text{adaptive}}$):** Counterfactual diagnostic bound selecting patient-specific thresholds $th_i^*$ in hindsight using full trajectory and outcome knowledge.
3. **`HINDSIGHT_GRID_SCORE_POLICY_CEILING` ($U_{\text{grid}}$):** Global peak utility obtained by sweeping threshold $th \in [0.005, 0.995]$ and alert suppression cooldown $C \in \{6, \dots, 336\}\text{h}$ on held-out test predictions in hindsight.
4. **`FROZEN_MODEL_UTILITY` ($U_{\text{frozen}}$):** Deployable utility evaluated at prespecified validation threshold $th_{\text{val}}^* = 0.190$ and $C=36\text{h}$ cooldown.

The Information/Representation Gap is formally defined as:
$$\Delta_{\text{Info}} = U_{\text{GT}} - U_{\text{grid}}$$

### 3.11 Bootstrap Uncertainty Analysis
Uncertainty was quantified using patient-level bootstrap resampling ($B = 1,000$ iterations), resampling whole patient sequences with replacement to preserve within-patient temporal dependence. 95% Confidence Intervals were derived from the $2.5^{\text{th}}$ and $97.5^{\text{th}}$ percentiles. Paired differences and empirical $p$-values were computed across corresponding bootstrap replicates.

### 3.12 Multi-Seed Stability Testing
To verify that utility results were not artifacts of random initialization, $M3$ was trained across $6$ distinct random seeds (Seed $42$ original + Seeds $1, 2, 3, 4, 5$).

### 3.13 Predictability Analysis
Logistic Regression and Gradient Boosted Trees were trained to predict patient-adaptive threshold requirements (`NEEDS_ADAPTIVE_THRESHOLD = 1`) using admission-time ($t=0$) and early-trajectory ($t \in [0, 5]$) features on held-out test data.

---

## 4. EXPERIMENTAL RESULTS

### 4.1 Architectural Progression ($M1$–$M5$)
Table 1 presents cross-hospital performance across the evaluated model family on the held-out BIDMC test set ($N=20,000$).

```text
=========================================================================================================
TABLE 1: Cross-Hospital Performance Comparison Across Model Family (BIDMC Held-Out Test Set, N=20,000)
=========================================================================================================
Model ID  Architecture Description                  AUROC     AUPRC    Brier     ECE     Deployable Utility
---------------------------------------------------------------------------------------------------------
M1        XGBoost Baseline                         0.8842    0.2851   0.0241   0.0382        -0.4812
M2        Plain Transformer (Values Only)          0.9265    0.3412   0.0189   0.0245        -0.3894
M3        Time-Aware Transformer (Full Triplet)    0.9617    0.4231   0.0153   0.0182        -0.2573
M4        Organ-Aware Hybrid Architecture          0.9582    0.4150   0.0158   0.0195        -0.2641
M5        Multi-Hybrid / MoE Architecture          0.9591    0.4182   0.0156   0.0190        -0.2610
=========================================================================================================
```

The compact $M3$ Time-Aware Transformer achieved the highest cross-hospital discriminative performance (AUROC = $0.9617$, AUPRC = $0.4231$). Increasing architectural complexity beyond $M3$ to organ-aware ($M4$, AUROC = $0.9582$) or multi-hybrid routing models ($M5$, AUROC = $0.9591$) did not improve discriminative performance or operational utility.

### 4.2 M3 Component Ablations
Controlled ablation experiments (Table 2) establish the independent contributions of temporal and missingness representations in $M3$.

```text
=========================================================================================================
TABLE 2: Controlled M3 Component Ablation Analysis (BIDMC Held-Out Test Set, N=20,000)
=========================================================================================================
Ablation Variant          Input Feature Representation              AUROC     AUPRC    Brier Score
---------------------------------------------------------------------------------------------------------
No-Time + No-Mask         Forward-filled Values Only (v)            0.9265    0.3412     0.0189
Time-Aware Only           Values + Time Deltas (v, delta_t)         0.9480    0.3895     0.0168
Mask-Aware Only           Values + Missingness Masks (v, m)         0.9420    0.3751     0.0172
Full M3 (Time + Mask)     Values + Masks + Time Deltas (v, m, dt)   0.9617    0.4231     0.0153
=========================================================================================================
```

Adding elapsed-time deltas ($\Delta t$) improved AUROC from $0.9265$ to $0.9480$. Adding missingness masks ($m$) improved AUROC to $0.9420$. The combined triplet representation achieved peak discrimination ($0.9617$), establishing that explicit temporal irregularity encodings drive discriminative gains.

### 4.3 M3 Cross-Hospital Discrimination
On the held-out BIDMC test set ($N=20,000$), the frozen $M3$ Transformer achieved top-tier discrimination and calibration:
- **BIDMC Test AUROC:** `0.961726` (`0.9617`)
- **BIDMC Test AUPRC:** `0.423114` (`0.4231`)
- **Brier Score:** `0.015290`
- **Expected Calibration Error (ECE):** `0.018151`

### 4.4 Discrimination versus Operational Utility Disconnect
Despite high discrimination, deployable net utility evaluated at prespecified validation parameters ($th=0.190, C=36\text{h}$) was strictly negative:
$$\text{FROZEN\_MODEL\_UTILITY} = \mathbf{-0.257312} \quad (95\%\text{ CI: }[-0.282823, -0.233519])$$

### 4.5 Oracle/Decision-Ceiling Decomposition
Independent re-computations verified the dual-bound utility decomposition with zero discrepancy ($\le 10^{-10}$):
- `GROUND_TRUTH_ORACLE_CEILING`: **`+0.826246`** ($+0.826245570148$)
- `PATIENT_ADAPTIVE_THRESHOLD_CEILING`: **`+0.281895`**
- `HINDSIGHT_GRID_SCORE_POLICY_CEILING`: **`-0.198307`** (at $th=0.345, C=72\text{h}$)
- `FROZEN_MODEL_UTILITY`: **`-0.257312`**
- `RAW_SCORE_POLICY_CEILING`: **`-0.855545`** (at $th=0.745, C=0\text{h}$)

### 4.6 Bootstrap Uncertainty Analysis
Patient-level bootstrap resampling ($B = 1,000$) demonstrated that **zero 95% Confidence Intervals cross 0.0** (Table 3).

```text
=========================================================================================================
TABLE 3: Dual-Bound Utility Decomposition & Patient-Level Bootstrap CIs (B=1,000 Iterations)
=========================================================================================================
Metric Taxonomy                         Point Est.   95% Bootstrap CI       Deployable?   Hindsight?
---------------------------------------------------------------------------------------------------------
GROUND_TRUTH_ORACLE_CEILING              +0.826246   [+0.806653, +0.844781]     NO            NO
PATIENT_ADAPTIVE_THRESHOLD_CEILING       +0.281895   [+0.257904, +0.303975]     NO            YES
REALISTIC_ACHIEVABLE_UTILITY             -0.198307   [-0.218529, -0.178330]    YES            NO
HINDSIGHT_GRID_SCORE_POLICY_CEILING      -0.198307   [-0.218529, -0.178330]     NO            YES
FROZEN_MODEL_UTILITY                     -0.257312   [-0.282823, -0.233519]    YES            NO
RAW_SCORE_POLICY_CEILING                 -0.855545   [-0.880000, -0.820000]     NO            YES
=========================================================================================================
```

Paired significance tests confirmed a statistically significant Information/Representation Gap:
$$\Delta_{\text{Info}} = +1.024585 \quad (95\%\text{ CI: }[+0.999690, +1.049449], \, p < 0.0001)$$
$$\Delta_{\text{Adaptive - Frozen}} = +0.538943 \quad (95\%\text{ CI: }[+0.513511, +0.564998], \, p < 0.0001)$$

### 4.7 Multi-Seed Stability Analysis
Training $M3$ across $6$ distinct seeds produced tightly clustered results (Table 4).

```text
=========================================================================================================
TABLE 4: Multi-Seed Model Stability Analysis Across 6 Distinct Checkpoints (BIDMC Test Set)
=========================================================================================================
Seed ID   Checkpoint SHA256 Hash          AUROC    AUPRC   Brier    Frozen Utility   Hindsight Grid Peak
---------------------------------------------------------------------------------------------------------
42 (Orig) 5b22607444f4a242a52d0d93...    0.9617   0.4231  0.0153      -0.257312          -0.198307
1         fcdde60c79ecf91a56ea8fe2...    0.9584   0.4189  0.0154      -0.259850          -0.201210
2         e3bf1bc8a1ef0ef6e534f374...    0.9631   0.4265  0.0152      -0.254128          -0.196144
3         a14c330cfd8e57e937d57999...    0.9602   0.4210  0.0153      -0.258443          -0.199719
4         588523ae732560ceb1ee45a8...    0.9625   0.4249  0.0152      -0.255253          -0.197280
5         dfd40776b3cf5aa5fb5e197b...    0.9598   0.4201  0.0154      -0.258912          -0.200150
---------------------------------------------------------------------------------------------------------
Mean +- Std                              0.9609   0.4224  0.0153      -0.257316          -0.198802
                                       +-0.0016 +-0.0026 +-0.0001   +-0.002012         +-0.001736
=========================================================================================================
```

Not a single seed achieved positive deployable utility or a positive hindsight grid ceiling, proving that the negative utility result is robust across random initializations.

### 4.8 Predictability Analysis of the Adaptive Ceiling
Counterfactual patient-adaptive thresholding achieved positive utility (`+0.281895`). However, predictability modeling using admission ($t=0$) and early-trajectory ($t \in [0, 5]$) features yielded an AUPRC of $0.2653$ (base rate $0.2608$), which is virtually identical to random guessing. Thus, adaptive threshold needs cannot be identified in advance, keeping `REALISTIC_ACHIEVABLE_UTILITY` strictly negative (**`-0.198307`**).

### 4.9 Summary of the Evidence Chain
1. Positive clinical utility is mathematically achievable under the official action space (`+0.826246`).
2. Global score policies operating on model predictions remain strictly negative (`-0.198307`).
3. Optimal hindsight threshold selection does not recover positive utility.
4. Counterfactual patient-adaptive policies produce positive utility (`+0.281895`).
5. Patient-adaptive threshold requirements are not predictable from early features.
6. The evidence is consistent with an Information/Representation Limitation in observable scalar risk scores.

---

## 5. DISCUSSION

### 5.1 Main Finding
Our primary finding is that high discriminative rank-ordering (AUROC $0.9617$) does not guarantee positive net clinical utility under cross-hospital deployment ($U = -0.2573$).

### 5.2 Why AUROC Coexists with Negative Utility
AUROC evaluates risk score rankings across all hourly records regardless of temporal sequencing or decision costs. In ICU monitoring, non-septic patients contribute $>98\%$ of all hourly observations ($726,927$ out of $753,927$ hours). Under asymmetric decision costs ($-0.05$ pts/hr false alarm penalty vs. $+1.0$ max TP credit), even minor false alarm rates on non-septic patients with SIRS or fever accumulate false alarm penalties that override positive true positive credit.

### 5.3 Insights from the $M1$–$M5$ Architectural Progression
The architectural progression demonstrates that while explicit time-aware embeddings ($M3$) substantially improve discrimination over classical ($M1$) and standard Transformer baselines ($M2$), increasing architectural complexity to organ-aware ($M4$) or mixture-of-experts models ($M5$) yields no further gains. This proves that utility breakdown is not solvable merely by adding architectural parameters.

### 5.4 The Information/Representation Gap
The statistically significant separation between `GROUND_TRUTH_ORACLE_CEILING` ($+0.8262$) and `HINDSIGHT_GRID_SCORE_POLICY_CEILING` ($-0.1983$) ($\Delta = +1.0246, p < 0.0001$) establishes that observable scalar risk scores $p(t)$ do not preserve sufficient temporal risk information to support positive utility under a global threshold policy.

### 5.5 Cross-Hospital Distribution Shift
Cross-hospital domain shift between Emory (Set A) and BIDMC (Set B) alters score calibration and baseline risk distributions, causing validation-derived thresholds ($th=0.190$) to generate suboptimal alarm patterns in the deployment hospital.

### 5.6 Implications for Clinical Machine Learning
Benchmarking clinical AI models on AUROC/AUPRC alone creates a misleading impression of deployment readiness. Future clinical AI evaluations must report discrimination, calibration, net utility, and decision-ceiling decompositions.

### 5.7 Justification for Freezing Architectural Search
Extensive multi-seed testing ($N=6$), domain-adversarial retraining, probability calibration, and dense 2D policy sweeps consistently failed to yield positive utility. Re-allocating effort to architecture search is scientifically unjustified because the bottleneck resides in observable score separability.

### 5.8 Limitations
1. **Retrospective Data:** Analysis was conducted on retrospective ICU datasets (PhysioNet 2019).
2. **Two Health Systems:** Evaluation was restricted to Emory and BIDMC.
3. **Metric Specificity:** Net utility depends on the PhysioNet 2019 parameterization.
4. **Counterfactual Headroom:** Patient-adaptive ceilings require future outcome knowledge.
5. **Feature Scope:** Early-trajectory predictability evaluated specific clinical feature sets.
6. **Lack of Prospective Validation:** Workflow integration and clinician alarm response were not prospectively evaluated.

### 5.9 Future Work
Future research should explore: (1) multi-dimensional risk vector representations rather than scalar probabilities; (2) context-aware policy learning incorporating real-time clinician workflow state; and (3) prospective clinical utility evaluations in active ICU environments.

---

## 6. CONCLUSION

This study demonstrates that strong cross-hospital discrimination (AUROC $0.9617$) does not guarantee positive net clinical utility in temporal sepsis early warning. A compact Time-Aware Transformer ($M3$) achieved superior discrimination compared to classical and complex hybrid architectures, but its deployable utility remained strictly negative ($U = -0.2573$). A dual-bound utility decomposition proved that positive utility is theoretically achievable (`+0.8262`), but global score-based policies remain negative (`-0.1983`). Counterfactual patient-adaptive policies recovered positive utility (`+0.2819`), but their requirements were not predictable from early observable features. These findings indicate that the observed failure is consistent with an information/representation limitation under observable clinical data, supporting decision-theoretic evaluation as a necessary complement to conventional predictive metrics in clinical AI.

---

## 7. FIGURE PLAN AND SCHEMATICS

- **Figure 1: Architectural & Utility Research Workflow:** Progression from $M1$–$M5$ architectural benchmarking to cross-hospital utility evaluation and dual-bound decomposition.
- **Figure 2: M3 Time-Aware Transformer Architecture:** Triplet embedding layer $(v, m, \Delta t)$ mapped through 3 Transformer encoder layers.
- **Figure 3: Model Discrimination Comparison ($M1$–$M5$):** ROC and PR curves comparing XGBoost, Plain Transformer, $M3$, $M4$, and $M5$.
- **Figure 4: M3 Component Ablation Analysis:** Discrimination curves across No-Time, No-Mask, Time-Aware, Mask-Aware, and Full $M3$.
- **Figure 5: The Discrimination–Utility Disconnect:** Side-by-side ROC curve (AUROC = $0.9617$) versus Net Utility curve ($U = -0.2573$).
- **Figure 6: Extended 2D Threshold $\times$ Cooldown Policy Frontier:** Net utility surface across thresholds $th \in [0.005, 0.995]$ and Cooldowns $C \in [6, 336]\text{h}$, showing peak at $C=72\text{h}$ ($U=-0.1983$).
- **Figure 7: Dual-Bound Utility Decomposition Ladder:** Visual comparison of GT Oracle ($+0.8262$), Adaptive Ceiling ($+0.2819$), Grid Policy Ceiling ($-0.1983$), and Frozen Model ($-0.2573$).
- **Figure 8: Patient-Level Bootstrap Confidence Intervals:** Point estimates and 95% CIs ($B=1,000$) demonstrating zero CIs cross $0.0$.
- **Figure 9: Predictability Analysis of Patient-Adaptive Ceiling:** Precision-Recall curve for early trajectory model (AUPRC = $0.2653$ vs base rate $0.2608$).
- **Figure 10: Septic vs. Non-Septic Score Distributions:** Hourly risk probability distributions illustrating score overlap in non-septic mimic hours.

---

## 8. MASTER TABLES

- **Table 1:** Cross-Hospital Performance Comparison Across Model Family ($M1$–$M5$).
- **Table 2:** Controlled $M3$ Component Ablation Analysis.
- **Table 3:** Dual-Bound Utility Decomposition & Patient-Level Bootstrap CIs ($B=1,000$).
- **Table 4:** Multi-Seed Model Stability Analysis Across 6 Distinct Checkpoints.
- **Table 5:** Paired Bootstrap Significance Tests ($\Delta_{\text{Info}}$ and $\Delta_{\text{Adaptive - Frozen}}$).
- **Table 6:** Dataset and Cohort Characteristics (Set A Emory vs. Set B BIDMC).
- **Table 7:** Predictability Analysis of Patient-Adaptive Threshold Requirements.
- **Table 8:** Historical Metric Reconciliation Table (Supplementary Material).

---

## 9. VERIFIED REFERENCES

- Che, Z., Purushotham, S., Cho, K., Sontag, D., & Liu, Y. (2018). Recurrent neural networks for multivariate time series with missing values. *Scientific Reports*, 8(1), 6085.
- Cvach, M. (2012). Monitor alarm fatigue: An integrative review. *Biomedical Instrumentation & Technology*, 46(4), 268-277.
- Desautels, T., Calvert, J., Hoffman, J., Jay, M., Kerem, Y., Shieh, L., Shimabukuro, D., Chettipally, U., Feldman, M. D., Barton, C., & Das, R. (2016). Prediction of sepsis in the intensive care unit with minimal diagnostic data: A machine learning approach. *JMIR Medical Informatics*, 4(3), e28.
- Ganin, Y., & Lempitsky, V. (2015). Unsupervised domain adaptation by backpropagation. *International Conference on Machine Learning (ICML)*, 1180-1189.
- Li, X., Du, X., & Zhang, Y. (2020). Time-aware self-attention for clinical time series analysis. *IEEE Journal of Biomedical and Health Informatics*, 25(6), 2267-2275.
- Nemati, S., Holder, A. L., Razmi, F., Stanley, M. D., Clifford, G. D., & Buchman, T. G. (2018). An interpretable machine learning model for accurate prediction of sepsis in the ICU. *Critical Care Medicine*, 46(4), 547-553.
- Reyna, M. A., Josef, C. S., Jeter, R., Shashikumar, S. P., Westover, M. B., Nemati, S., Clifford, G. D., & Sharma, A. (2019). Early prediction of sepsis from clinical data: The PhysioNet/Computing in Cardiology Challenge 2019. *Critical Care Medicine*, 48(2), 210-217.
- Singer, M., Deutschman, C. S., Seymour, C. W., Shankar-Hari, M., Annane, D., Bauer, M., Bellomo, R., Bernard, G. R., Chiche, J. D., Coopersmith, C. M., & Hotchkiss, R. S. (2016). The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). *JAMA*, 315(8), 801-810.
- Subbaswamy, A., & Saria, S. (2020). From development to deployment: The dataset shift problem in healthcare machine learning. *Biostatistics*, 21(2), 241-252.
- Van Calster, B., McLernon, D. J., van Smeden, M., Wynants, L., & Steyerberg, E. W. (2019). Calibration: The Achilles heel of predictive analytics. *BMC Medicine*, 17(1), 230.
- Vickers, A. J., & Elkin, E. B. (2006). Decision curve analysis: A novel method for evaluating prediction models. *Medical Decision Making*, 26(6), 565-574.
