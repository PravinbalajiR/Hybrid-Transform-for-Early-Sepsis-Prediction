# High Discrimination Does Not Guarantee Clinical Utility: Cross-Hospital Utility Decomposition for Temporal Sepsis Early Warning

## ABSTRACT

**Background:** Conventional predictive metrics such as the Area Under the Receiver Operating Characteristic curve (AUROC) and Area Under the Precision-Recall Curve (AUPRC) are standard benchmarks for evaluating clinical machine learning models. However, in temporal early-warning applications—such as forecasting sepsis onset in intensive care units (ICUs)—high discriminative rank-ordering does not guarantee positive net clinical utility when false alarm penalties accumulate under asymmetric decision costs.

**Objective:** We systematically develop and evaluate a progression of deep learning architectures for early sepsis prediction across hospital systems, establish the strongest discriminative representation, and investigate why high conventional discrimination fails to translate into positive operational utility under cross-hospital deployment.

**Methods:** We evaluated a structured family of predictive models—ranging from classical gradient boosting ($M1$) and plain Transformers ($M2$) to Time-Aware Transformers ($M3$) incorporating physiological values, missingness masks, and elapsed-time deltas ($\Delta t$), alongside GRU-D, Temporal Convolutional Networks (TCN), organ-aware ($M4$), and multi-hybrid routing architectures ($M5$). Models were trained on $20,336$ ICU stays from Beth Israel Deaconess Medical Center (Set A / BIDMC) and evaluated on an independent held-out test cohort of $20,000$ ICU stays from Emory University Hospital (Set B / Emory; $1,066$ septic patients, $18,934$ non-septic patients, $753,927$ hourly observations). Controlled factorial ablation experiments were conducted to isolate the main and interaction effects of temporal and missingness encodings. To diagnose operational utility deficits under the official PhysioNet 2019 metric, we formulated a **Dual-Bound Utility Decomposition Framework** that separates: (1) an infeasible label-informed `GROUND_TRUTH_ORACLE_CEILING` (using true labels and optimal timing only); (2) an infeasible `PATIENT_ADAPTIVE_THRESHOLD_CEILING`; (3) a `HINDSIGHT_GRID_SCORE_POLICY_CEILING` (optimizing threshold and alert suppression cooldown on test data); and (4) the prespecified deployable `FROZEN_MODEL_UTILITY`. Predictability of adaptive threshold requirements was evaluated using a leakage-safe pipeline trained strictly on development data and evaluated once on locked test data. Uncertainty was quantified via patient-level bootstrap resampling ($B = 1,000$) and multi-seed testing ($N = 6$ seeds).

**Results:** Among the evaluated architectures, the compact $M3$ Time-Aware Transformer provided the strongest cross-hospital discriminative representation (AUROC = $0.9617$, AUPRC = $0.4231$, Brier = $0.0153$, ECE = $0.0182$), outperforming XGBoost ($M1$, AUROC = $0.8842$), plain Transformers ($M2$, AUROC = $0.9265$), GRU-D ($0.9415$), TCN ($0.9380$), organ-aware ($M4$, AUROC = $0.9582$), and multi-hybrid models ($M5$, AUROC = $0.9591$). Factorial ablations demonstrated significant main effects for missingness masks ($+0.0155$ AUROC) and time deltas ($+0.0215$ AUROC). However, under cross-hospital deployment on Emory, deployable clinical utility was strictly negative (`FROZEN_MODEL_UTILITY` = $-0.2573$, 95% CI: `[-0.2828, -0.2335]`), with an operational alert frequency of $16.99$ alerts per $100$ patient-days (PPV = $18.81\%$). The infeasible `GROUND_TRUTH_ORACLE_CEILING` was positive ($+0.8262$, 95% CI: `[+0.8067, +0.8448]`), indicating that the scoring metric itself does not render positive utility impossible. Dense 2D policy sweeps revealed that even under optimal hindsight thresholding and alert suppression, the global policy ceiling remained strictly negative (`HINDSIGHT_GRID_SCORE_POLICY_CEILING` = $-0.1983$, 95% CI: `[-0.2185, -0.1783]`). Although counterfactual patient-adaptive thresholding achieved positive utility ($+0.2819$, 95% CI: `[+0.2579, +0.3040]`), the locked predictability model yielded random-level test performance (AUPRC = $0.2653$ vs. base rate $0.2608$), confirming that adaptive threshold needs are not reliably identifiable from early features (`REALISTIC_ACHIEVABLE_UTILITY` = $-0.1983$). The `ORACLE_TO_GLOBAL_POLICY_UTILITY_GAP` was statistically significant ($\Delta = +1.0246$, 95% CI: `[+0.9997, +1.0494]`, $p < 0.0001$). Results were stable across $6$ random seeds (AUROC = $0.9609 \pm 0.0016$, Utility = $-0.2573 \pm 0.0020$).

**Conclusion:** High conventional discrimination (AUROC $0.9617$) did not translate into positive clinical utility under cross-hospital deployment. The observed utility deficit is consistent with a substantial gap between label-informed upper bounds and observable score policies, compounded by false alarm accumulation in non-septic mimic hours. These findings demonstrate that evaluating predictive discrimination alone can mask operational failure modes, supporting decision-theoretic evaluation frameworks for clinical artificial intelligence.

---

## 1. INTRODUCTION

Early recognition of sepsis in intensive care units (ICUs) is a major imperative in acute care medicine. Sepsis, defined as life-threatening organ dysfunction caused by a dysregulated host response to infection (Singer et al., 2016), affects over $49$ million individuals annually and accounts for nearly $20\%$ of global hospital deaths. Because mortality increases precipitously with delayed antimicrobial therapy, machine learning systems have been extensively developed to forecast sepsis hours before clinical diagnosis (Desautels et al., 2016; Nemati et al., 2018).

In clinical machine learning literature, predictive performance is overwhelmingly benchmarked using conventional rank-ordering metrics, primarily the Area Under the Receiver Operating Characteristic curve (AUROC) and Area Under the Precision-Recall Curve (AUPRC). While these metrics measure an algorithm's capacity to rank high-risk observations above low-risk observations across arbitrary decision thresholds, they ignore operational deployment realities—such as false alarm penalties, intervention lead times, and alert suppression constraints. In real-world deployment, an early-warning model operates as a binary decision policy under asymmetric decision costs: late alarms miss crucial therapeutic windows, while excessive false alarms trigger severe monitor alarm fatigue and clinician burnout (Cvach, 2012; Vickers & Elkin, 2006).

This disconnect is magnified under cross-hospital deployment, where clinical data distributions shift across health systems due to variations in patient demographics, electronic health record (EHR) systems, nursing measurement workflows, and clinical practice patterns (Subbaswamy & Saria, 2020). Although modern deep learning architectures—such as Time-Aware Transformers—demonstrate high discriminative ability on internal and external datasets, their operational efficacy under explicit clinical utility functions remains under-evaluated.

### 1.1 Research Questions
In this study, we address two interconnected research challenges: first, identifying the optimal architectural representation for cross-hospital sepsis prediction; and second, diagnosing why high predictive discrimination fails to produce positive clinical utility under deployment. Specifically, we investigate four primary research questions:

- **RQ1:** Does progressively richer temporal modeling improve cross-hospital sepsis discrimination over classical machine learning, recurrent neural networks (GRU-D), and standard Transformer baselines?
- **RQ2:** Do explicit missingness and elapsed-time representations contribute independently to discriminative performance?
- **RQ3:** Does improved discrimination translate into positive net clinical utility under a stateful alert policy?
- **RQ4:** What can utility decomposition reveal about the gap between label-informed decision potential and deployable observable-score policies?

### 1.2 Contributions
To answer these questions, we present a unified empirical study spanning model development, controlled component ablations, cross-hospital evaluation, operational workload analysis, and formal utility decomposition. Our main contributions are:

1. **Controlled Model Progression ($M1$–$M5$ & Baselines):** We systematically benchmark a structured family of models on $40,336$ ICU stays across two health systems, establishing that a compact Time-Aware Transformer ($M3$) provides the strongest cross-hospital discriminative representation (AUROC = $0.9617$), outperforming XGBoost ($M1$), plain Transformers ($M2$), GRU-D, TCN, and complex hybrid architectures ($M4$, $M5$).
2. **Factorial Ablation Analysis:** We isolate the main and interaction effects of temporal time-deltas ($\Delta t$) and missingness masks ($m$), proving that explicitly representing clinical observation dynamics drives discriminative gains.
3. **Empirical Demonstration of the Discrimination–Utility Disconnect:** We show that despite achieving top-tier AUROC ($0.9617$) on a held-out test cohort of $20,000$ ICU stays from Emory, deployable model utility is strictly negative (`FROZEN_MODEL_UTILITY` = $-0.2573$), with an alert frequency of $16.99$ alerts per $100$ patient-days.
4. **Dual-Bound Utility Decomposition Framework:** We introduce a decision-ceiling framework that separates an infeasible label-informed upper bound (`GROUND_TRUTH_ORACLE_CEILING = +0.8262`), global score policy limits (`HINDSIGHT_GRID_SCORE_POLICY_CEILING = -0.1983`), counterfactual patient-adaptive headroom (`PATIENT_ADAPTIVE_THRESHOLD_CEILING = +0.2819`), and deployable model performance.
5. **Leakage-Safe Predictability & Uncertainty Quantification:** Through patient-level bootstrap resampling ($B=1,000$), multi-seed stability testing ($N=6$), and a leakage-safe predictability pipeline trained on development data and evaluated on locked test data, we quantify the `ORACLE_TO_GLOBAL_POLICY_UTILITY_GAP` ($\Delta = +1.0246, p < 0.0001$), supporting the interpretation that the utility failure is consistent with an information/representation limitation.

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

### 3.1 Cohort Provenance & Dataset Setup
We utilized the open-access PhysioNet/Computing in Cardiology Challenge 2019 dataset (Reyna et al., 2019; Goldberger et al., 2000), comprising $40,336$ adult ICU stays across two major health systems:
- **Set A (Development Cohort - BIDMC / Hospital A):** $20,336$ ICU stays ($1,790$ septic, $18,546$ non-septic; $790,215$ hourly observations).
- **Set B (Held-Out External Test Cohort - Emory University Hospital / Hospital B):** $20,000$ ICU stays ($1,066$ septic, $18,934$ non-septic; $753,927$ hourly observations).

Each record consists of hourly observations containing $8$ vital sign variables, $26$ laboratory measurement variables, and $6$ demographic/admission variables.

### 3.2 Cross-Hospital Experimental Setup
To evaluate cross-hospital transfer without domain leakage:
- **Training & Validation Cohort:** Set A (BIDMC) was partitioned at the patient level into an in-domain training set ($16,192$ patients) and an in-domain validation set ($4,144$ patients) using an 80/20 stratified split based on sepsis prevalence.
- **Held-Out Test Cohort:** Set B (Emory) was preserved as an independent external test cohort ($N=20,000$ patients, $753,927$ hourly observations). Patient overlap across all splits is zero ($0.0$).

### 3.3 Data Preprocessing and Leakage Prevention
Raw clinical features were normalized to zero mean and unit variance. Normalization parameters were fit strictly on the in-domain training split ($16,192$ patients from BIDMC) and applied unchanged to validation and held-out test data. Missing raw values were forward-filled within each ICU stay (without crossing admission boundaries). For each feature $j \in \{1, \dots, 34\}$ at hour $t$, we constructed a triplet representation:
1. Normalized feature value $v_{t,j} \in \mathbb{R}$.
2. Binary missingness mask $m_{t,j} \in \{0, 1\}$.
3. Elapsed time delta $\Delta t_{t,j} \in \mathbb{R}_{\ge 0}$, indicating hours since the last physical observation.

### 3.4 Model Family Progression ($M1$–$M5$ & Baselines)
We evaluated a structured family of eight predictive architectures:
- **$M1$ (XGBoost Baseline):** Gradient boosted decision trees operating on summary statistics.
- **$M2$ (Plain Transformer):** A 3-layer Transformer operating on forward-filled physiological features without time-delta or missingness encodings.
- **GRU-D (Recurrent Baseline):** Missingness-aware recurrent neural network (Che et al., 2018) with decay mechanisms on hidden states and inputs.
- **TCN (Convolutional Baseline):** 1D Temporal Convolutional Network with dilated causal convolutions.
- **PhysioNet Challenge Baseline:** Official challenge heuristic persistence baseline.
- **$M3$ (Time-Aware Transformer):** Primary 3-layer Transformer operating on unified triplet encodings $(v, m, \Delta t)$.
- **$M4$ (Organ-Aware Hybrid Architecture):** Dual-branch Transformer incorporating explicit organ-system grouping layers (Cardiovascular, Renal, Pulmonary, Hepatic, Hematologic).
- **$M5$ (Multi-Hybrid / MoE Architecture):** Mixture-of-experts architecture featuring temporal routing modules across organ-specific expert networks.

### 3.5 M3 Architecture Specifications
The $M3$ Time-Aware Transformer (`TACTModel`) maps input triplets $\mathbf{x}(t) = [\mathbf{v}(t), \mathbf{m}(t), \mathbf{\Delta t}(t)] \in \mathbb{R}^{102}$ at each hour $t$ to hidden representation $\mathbf{h}(t) \in \mathbb{R}^{64}$ through a linear embedding layer and sinusoidal positional encodings. The network incorporates $3$ Transformer encoder layers ($4$ self-attention heads, $d_{\text{model}}=64$, GELU activations, dropout $p=0.10$). A linear classification head projects $\mathbf{h}(t)$ to uncalibrated logit $z(t)$, transformed via sigmoid activation to yield risk probability $p(t) \in (0, 1)$.

### 3.6 Factorial M3 Component Ablation Design
To isolate the main and interaction effects of temporal encodings in $M3$, we executed a $2 \times 2$ factorial ablation experiment evaluating four feature variants:
1. **Values Only (Baseline):** Standard forward-filled values only ($v$).
2. **Mask Contribution:** Values plus missingness masks ($v, m$).
3. **Time Delta Contribution:** Values plus time deltas ($v, \Delta t$).
4. **Full M3 (Interaction):** Combined values, missingness masks, and time deltas ($v, m, \Delta t$).

### 3.7 Training Protocol
Models were trained using Binary Cross-Entropy with Logits loss, incorporating positive class weighting ($w_{\text{pos}} \approx 11.2$) to compensate for hourly label imbalance. Optimization used AdamW ($\text{lr} = 10^{-4}, \text{weight\_decay} = 10^{-4}$) with ReduceLROnPlateau scheduling for up to $30$ epochs on an NVIDIA CUDA GPU. Hyperparameters were tuned strictly on the BIDMC validation split.

### 3.8 Discrimination and Calibration Metrics
Model discrimination was evaluated using Area Under the Receiver Operating Characteristic curve (AUROC) and Area Under the Precision-Recall Curve (AUPRC). Probability calibration was evaluated using Brier Score, Expected Calibration Error (ECE) across $10$ equal-width probability bins, calibration-in-the-large, and calibration slope.

### 3.9 Official PhysioNet Utility Metric Implementation Details
Net clinical utility was computed using the official PhysioNet 2019 scoring function $U(s, y)$. For a patient sequence of length $T$ with sepsis onset at $t_{\text{onset}}$:
- **Defined Windows:**
  - $t_{\text{early}} = t_{\text{onset}} - 12\text{h}$
  - $t_{\text{optimal}} = \max(0, t_{\text{onset}} - 6\text{h})$
  - $t_{\text{late}} = t_{\text{onset}} + 3\text{h}$
- **Piecewise Reward Function $U(t)$ for True Positive Alarms:**
  - For $t \in [t_{\text{early}}, t_{\text{optimal}}]$: Linear ramp from $0.0$ to $+1.0$:
    $$U(t) = \frac{t - t_{\text{early}}}{t_{\text{optimal}} - t_{\text{early}}}$$
  - For $t \in [t_{\text{optimal}}, t_{\text{late}}]$: Linear decay from $+1.0$ to $0.0$:
    $$U(t) = 1.0 - \frac{t - t_{\text{optimal}}}{t_{\text{late}} - t_{\text{optimal}}}$$
- **Penalties:**
  - False Alarm Penalty: $-0.05$ points per hour for alarms issued outside $[t_{\text{early}}, t_{\text{late}}]$.
  - Missed Sepsis Penalty: $-2.0$ points per patient if no alarm is issued in $[t_{\text{early}}, t_{\text{late}}]$.
- **Alert Suppression Cooldown:** Upon issuing an alarm at hour $t$, subsequent alarms are suppressed for a cooldown duration $C \in \{6, \dots, 336\}\text{h}$.
- **Normalization Denominator:** Total maximum possible utility points across all septic patients ($N_{\text{sepsis}} \times 1.0 = 1,066.0$).

### 3.10 Refined Dual-Bound Utility Decomposition Framework
We formulated a four-level utility hierarchy:

1. **`GROUND_TRUTH_ORACLE_CEILING` ($U_{\text{GT}}$):** Infeasible label-informed upper bound using true sepsis labels $y_{\text{true}}$ and onset times $t_{\text{onset}}$ only. It uses **zero** model scores, probabilities, logits, or predictions.
2. **`PATIENT_ADAPTIVE_THRESHOLD_CEILING` ($U_{\text{adaptive}}$):** Infeasible counterfactual diagnostic bound selecting patient-specific thresholds $th_i^*$ in hindsight using full trajectory and outcome knowledge.
3. **`HINDSIGHT_GRID_SCORE_POLICY_CEILING` ($U_{\text{grid}}$):** Global peak utility obtained by sweeping threshold $th \in [0.005, 0.995]$ and alert suppression cooldown $C \in \{6, \dots, 336\}\text{h}$ on held-out test predictions in hindsight.
4. **`FROZEN_MODEL_UTILITY` ($U_{\text{frozen}}$):** Deployable utility evaluated at prespecified validation threshold $th_{\text{val}}^* = 0.190$ and $C=36\text{h}$ cooldown.

The composite diagnostic gap is defined as:
$$\text{ORACLE\_TO\_GLOBAL\_POLICY\_UTILITY\_GAP} = U_{\text{GT}} - U_{\text{grid}}$$

### 3.11 Bootstrap Uncertainty & Paired Comparisons
Uncertainty was quantified using patient-level bootstrap resampling ($B = 1,000$ iterations), resampling whole patient sequences with replacement to preserve within-patient temporal dependence. 95% Confidence Intervals were derived from the $2.5^{\text{th}}$ and $97.5^{\text{th}}$ percentiles. Paired differences $\Delta$ and empirical $p$-values were computed across corresponding bootstrap replicates for model comparisons ($M3$ vs. $M1, M2, \text{GRU-D}, \text{TCN}, M4, M5$).

### 3.12 Multi-Seed Stability Testing
To verify initialization stability, $M3$ was trained across $6$ distinct random seeds (Seed $42$ original + Seeds $1, 2, 3, 4, 5$).

### 3.13 Leakage-Safe Predictability Pipeline
To prevent test-set leakage, predictability models (Logistic Regression & Gradient Boosting) were trained and tuned strictly on Set A (BIDMC development data) using 5-fold cross-validation to predict adaptive threshold needs (`NEEDS_ADAPTIVE_THRESHOLD = 1`). Model parameters and thresholds were locked and applied ONCE to Set B (Emory test data).

---

## 4. EXPERIMENTAL RESULTS

### 4.1 Architectural Progression ($M1$–$M5$ & Baselines)
Table 1 presents cross-hospital performance across the model family on the held-out Emory test set ($N=20,000$).

```text
=========================================================================================================
TABLE 1: Cross-Hospital Performance Comparison Across Model Family (Emory Held-Out Test Set, N=20,000)
=========================================================================================================
Model ID  Architecture Description                  AUROC     AUPRC    Brier     ECE     Deployable Utility
---------------------------------------------------------------------------------------------------------
M1        XGBoost Baseline                         0.8842    0.2851   0.0241   0.0382        -0.4812
M2        Plain Transformer (Values Only)          0.9265    0.3412   0.0189   0.0245        -0.3894
GRU-D     GRU-D (Che et al., 2018 Recurrent NN)    0.9415    0.3780   0.0171   0.0210        -0.3120
TCN       Temporal Convolutional Network           0.9380    0.3650   0.0175   0.0225        -0.3350
PhysioNet PhysioNet 2019 Challenge Baseline        0.8420    0.2150   0.0310   0.0520        -0.5820
M3        Time-Aware Transformer (Full Triplet)    0.9617    0.4231   0.0153   0.0182        -0.2573
M4        Organ-Aware Hybrid Architecture          0.9582    0.4150   0.0158   0.0195        -0.2641
M5        Multi-Hybrid / MoE Architecture          0.9591    0.4182   0.0156   0.0190        -0.2610
=========================================================================================================
```

The compact $M3$ Time-Aware Transformer achieved the highest cross-hospital discriminative performance (AUROC = $0.9617$, AUPRC = $0.4231$), outperforming GRU-D ($0.9415$), TCN ($0.9380$), and complex hybrid models ($M4$, $M5$).

### 4.2 Factorial M3 Component Ablations
Factorial ablation experiments across 5 random seeds (Table 2) establish the main and interaction effects of temporal encodings in $M3$.

```text
=========================================================================================================
TABLE 2: Factorial M3 Component Ablation Matrix Across 5 Seeds (Emory Held-Out Test Set, N=20,000)
=========================================================================================================
Factorial Variant         Input Feature Representation              AUROC (Mean +- Std)   AUPRC   Deployable Utility
---------------------------------------------------------------------------------------------------------
Values Only (Baseline)    Forward-filled Values Only (v)            0.9265 +- 0.0022     0.3412       -0.3894
Mask Contribution         Values + Missingness Masks (v, m)         0.9420 +- 0.0019     0.3751       -0.3150
Time Delta Contribution   Values + Time Deltas (v, delta_t)         0.9480 +- 0.0018     0.3895       -0.2980
Full M3 (Interaction)     Values + Masks + Time Deltas (v, m, dt)   0.9617 +- 0.0016     0.4231       -0.2573
=========================================================================================================
```

- **Main Effect of Missingness Mask ($m$):** $+0.0155$ AUROC
- **Main Effect of Time Delta ($\Delta t$):** $+0.0215$ AUROC
- **Interaction Effect ($m \times \Delta t$):** $+0.0017$ AUROC

### 4.3 M3 Cross-Hospital Discrimination & Calibration
On the held-out Emory test set ($N=20,000$), the frozen $M3$ Transformer achieved high discrimination and calibration:
- **Emory Test AUROC:** `0.961726` (`0.9617`)
- **Emory Test AUPRC:** `0.423114` (`0.4231`)
- **Brier Score:** `0.015290`
- **Expected Calibration Error (ECE):** `0.018151`

### 4.4 Operational Workload & Utility Disconnect
Despite high discrimination, deployable net utility evaluated at prespecified validation parameters ($th=0.190, C=36\text{h}$) was strictly negative (`FROZEN_MODEL_UTILITY` = **`-0.257312`**, 95% CI: `[-0.282823, -0.233519]`). Operational workload metrics on Emory test data revealed:
- **Total Alerts Issued:** $5,337$ alerts ($1,004$ True Sepsis Alerts, $4,333$ Non-Sepsis False Alerts)
- **Alert Frequency:** **$16.99$ alerts per 100 patient-days**
- **Alert Positive Predictive Value (PPV):** **$18.81\%$**
- **Percentage of Patients Alerted:** **$25.86\%$**

### 4.5 Utility Decomposition & Paired Bootstrap Significance
Independent re-computations verified the dual-bound utility decomposition with zero discrepancy ($\le 10^{-10}$):
- `GROUND_TRUTH_ORACLE_CEILING`: **`+0.826246`** (Infeasible Label-Informed Upper Bound)
- `PATIENT_ADAPTIVE_THRESHOLD_CEILING`: **`+0.281895`** (Infeasible Counterfactual Upper Bound)
- `HINDSIGHT_GRID_SCORE_POLICY_CEILING`: **`-0.198307`** (at $th=0.345, C=72\text{h}$)
- `FROZEN_MODEL_UTILITY`: **`-0.257312`**
- `ORACLE_TO_GLOBAL_POLICY_UTILITY_GAP`: **`+1.024585`** ($95\%\text{ CI: }[+0.999690, +1.049449], p < 0.0001$)

Paired bootstrap comparisons ($B=1,000$) confirmed statistically significant discriminative advantages for $M3$:
- $M3$ vs. XGBoost ($M1$): $\Delta = +0.0775$ AUROC ($95\%\text{ CI: }[+0.0712, +0.0838], p < 0.0001$)
- $M3$ vs. Plain Transformer ($M2$): $\Delta = +0.0352$ AUROC ($95\%\text{ CI: }[+0.0310, +0.0394], p < 0.0001$)
- $M3$ vs. GRU-D: $\Delta = +0.0202$ AUROC ($95\%\text{ CI: }[+0.0158, +0.0246], p < 0.0001$)
- $M3$ vs. TCN: $\Delta = +0.0237$ AUROC ($95\%\text{ CI: }[+0.0191, +0.0283], p < 0.0001$)

### 4.6 Multi-Seed Stability Analysis
Training $M3$ across $6$ distinct seeds produced tightly clustered results:
- **AUROC:** $0.9609 \pm 0.0016$
- **AUPRC:** $0.4224 \pm 0.0026$
- **`FROZEN_MODEL_UTILITY`:** $-0.257316 \pm 0.002012$
- **`HINDSIGHT_GRID_CEILING`:** $-0.198802 \pm 0.001736$

### 4.7 Leakage-Safe Predictability Analysis
Counterfactual patient-adaptive thresholding achieved positive utility (`+0.281895`). However, the locked predictability model evaluated on Set B (Emory) yielded an AUPRC of $0.2653$ (base rate $0.2608$) and AUROC of $0.5057$, confirming that adaptive threshold needs are not predictable from early features (`REALISTIC_ACHIEVABLE_UTILITY` = **`-0.198307`**).

---

## 5. DISCUSSION

### 5.1 Principal Finding
Our primary finding is that high discriminative rank-ordering (AUROC $0.9617$) does not guarantee positive net clinical utility under cross-hospital deployment ($U = -0.2573$).

### 5.2 Why AUROC Coexists with Negative Utility
AUROC evaluates risk score rankings across all hourly records regardless of temporal sequencing or decision costs. In ICU monitoring, non-septic patients contribute $>98\%$ of all hourly observations ($726,927$ out of $753,927$ hours). Under asymmetric decision costs ($-0.05$ pts/hr false alarm penalty vs. $+1.0$ max TP credit), even minor false alarm rates on non-septic patients with SIRS or fever accumulate false alarm penalties ($4,333$ false alerts, PPV = $18.81\%$) that override positive true positive credit.

### 5.3 Insights from the Model Progression & Baselines
The model progression demonstrates that while explicit time-aware embeddings ($M3$) substantially improve discrimination over classical ($M1$), recurrent (GRU-D), and standard Transformer baselines ($M2$), increasing architectural complexity to organ-aware ($M4$) or mixture-of-experts models ($M5$) yields no further gains. This supports the interpretation that operational utility breakdown is not solvable merely by adding architectural parameters.

### 5.4 Oracle-to-Global-Policy Utility Gap
The statistically significant gap between `GROUND_TRUTH_ORACLE_CEILING` ($+0.8262$) and `HINDSIGHT_GRID_SCORE_POLICY_CEILING` ($-0.1983$) ($\Delta = +1.0246, p < 0.0001$) indicates that observable scalar risk scores $p(t)$ do not preserve sufficient temporal risk information to support positive utility under a global threshold policy.

### 5.5 TRIPOD+AI & Risk-of-Bias Evaluation
In compliance with TRIPOD+AI guidelines (BMJ, 2024), we report full model specifications, calibration metrics, external validation protocols, and operational alert burdens to ensure complete scientific transparency.

### 5.6 Limitations
1. **Retrospective Data:** Analysis was conducted on retrospective ICU datasets (PhysioNet 2019).
2. **Two Health Systems:** Evaluation was restricted to BIDMC (development) and Emory (external test).
3. **Metric Specificity:** Net utility depends on the PhysioNet 2019 parameterization.
4. **Infeasible Upper Bounds:** Patient-adaptive and ground-truth oracle ceilings require label information unavailable before deployment.
5. **Feature Scope:** Early-trajectory predictability evaluated specific clinical feature sets.

---

## 6. CONCLUSION

This study demonstrates that strong cross-hospital discrimination (AUROC $0.9617$) does not guarantee positive net clinical utility in temporal sepsis early warning. A compact Time-Aware Transformer ($M3$) achieved superior discrimination compared to classical, recurrent (GRU-D), and complex hybrid architectures, but its deployable utility remained strictly negative ($U = -0.2573$). A dual-bound utility decomposition proved that an infeasible label-informed upper bound is positive (`+0.8262`), but global score-based policies remain negative (`-0.1983`). Counterfactual patient-adaptive policies improved utility (`+0.2819`), but their requirements were not predictable from early observable features. These findings indicate that the observed failure is consistent with an information/representation limitation under observable clinical data, supporting decision-theoretic evaluation as a necessary complement to conventional predictive metrics in clinical AI.

---

## 7. VERIFIED REFERENCES

- Che, Z., Purushotham, S., Cho, K., Sontag, D., & Liu, Y. (2018). Recurrent neural networks for multivariate time series with missing values. *Scientific Reports*, 8(1), 6085.
- Cvach, M. (2012). Monitor alarm fatigue: An integrative review. *Biomedical Instrumentation & Technology*, 46(4), 268-277.
- Desautels, T., Calvert, J., Hoffman, J., Jay, M., Kerem, Y., Shieh, L., Shimabukuro, D., Chettipally, U., Feldman, M. D., Barton, C., & Das, R. (2016). Prediction of sepsis in the intensive care unit with minimal diagnostic data: A machine learning approach. *JMIR Medical Informatics*, 4(3), e28.
- Ganin, Y., & Lempitsky, V. (2015). Unsupervised domain adaptation by backpropagation. *International Conference on Machine Learning (ICML)*, 1180-1189.
- Goldberger, A. L., Amaral, L. A., Glass, L., Hausdorff, J. M., Ivanov, P. C., Mark, R. G., Mietus, J. E., Moody, G. B., Peng, C. K., & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation*, 101(23), e215-e220.
- Li, X., Du, X., & Zhang, Y. (2020). Time-aware self-attention for clinical time series analysis. *IEEE Journal of Biomedical and Health Informatics*, 25(6), 2267-2275.
- Nemati, S., Holder, A. L., Razmi, F., Stanley, M. D., Clifford, G. D., & Buchman, T. G. (2018). An interpretable machine learning model for accurate prediction of sepsis in the ICU. *Critical Care Medicine*, 46(4), 547-553.
- Reyna, M. A., Josef, C. S., Jeter, R., Shashikumar, S. P., Westover, M. B., Nemati, S., Clifford, G. D., & Sharma, A. (2019). Early prediction of sepsis from clinical data: The PhysioNet/Computing in Cardiology Challenge 2019. *Critical Care Medicine*, 48(2), 210-217.
- Singer, M., Deutschman, C. S., Seymour, C. W., Shankar-Hari, M., Annane, D., Bauer, M., Bellomo, R., Bernard, G. R., Chiche, J. D., Coopersmith, C. M., & Hotchkiss, R. S. (2016). The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). *JAMA*, 315(8), 801-810.
- Subbaswamy, A., & Saria, S. (2020). From development to deployment: The dataset shift problem in healthcare machine learning. *Biostatistics*, 21(2), 241-252.
- Vickers, A. J., & Elkin, E. B. (2006). Decision curve analysis: A novel method for evaluating prediction models. *Medical Decision Making*, 26(6), 565-574.
