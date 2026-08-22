# High Discrimination Does Not Guarantee Clinical Utility: An Oracle Decomposition of Cross-Hospital Sepsis Early Warning

## ABSTRACT

**Background:** Conventional discrimination metrics such as AUROC and AUPRC are standard benchmarks for evaluating clinical predictive models. However, in temporal early-warning applications—such as predicting sepsis onset in intensive care units (ICUs)—high discriminative ability does not guarantee positive net clinical utility when false alarm penalties accumulate under asymmetric decision costs.

**Objective:** We evaluate whether high predictive discrimination achieved by a Time-Aware Transformer model ($M3$) transfers across hospital systems to produce positive clinical utility under the official PhysioNet/CinC Challenge 2019 Utility Metric. Furthermore, we formulate a decision-ceiling decomposition framework to isolate the sources of performance breakdown under cross-hospital deployment.

**Methods:** We trained a Time-Aware Transformer model ($M3$) on $20,336$ ICU stays from Emory University Hospital (Set A) and evaluated its cross-hospital performance on a held-out test cohort of $20,000$ ICU stays from Beth Israel Deaconess Medical Center (Set B; $1,066$ septic patients, $18,934$ non-septic patients, $753,927$ hourly records). Model discrimination (AUROC, AUPRC) and calibration (Brier score, ECE) were evaluated alongside deployable net utility. To diagnose utility deficits, we decomposed performance into: (1) a perfect-information `GROUND_TRUTH_ORACLE_CEILING` (using true labels and optimal timing only); (2) a counterfactual `PATIENT_ADAPTIVE_THRESHOLD_CEILING`; (3) a `HINDSIGHT_GRID_SCORE_POLICY_CEILING` (optimizing threshold and alert suppression cooldown on test data); and (4) the prespecified deployable `FROZEN_MODEL_UTILITY`. Uncertainty was quantified via patient-level bootstrap resampling ($B = 1,000$) and multi-seed stability checks ($N = 6$ seeds).

**Results:** Under cross-hospital deployment on BIDMC, the frozen $M3$ Transformer achieved high discriminative performance (AUROC = $0.9617$, AUPRC = $0.4231$, Brier = $0.0153$, ECE = $0.0182$). However, deployable clinical utility was strictly negative (`FROZEN_MODEL_UTILITY` = $-0.2573$, 95% CI: `[-0.2828, -0.2335]`). The `GROUND_TRUTH_ORACLE_CEILING` was positive ($+0.8262$, 95% CI: `[+0.8067, +0.8448]`), proving that positive utility is mathematically achievable under the official action space. Extended 2D policy sweeps revealed that even under optimal hindsight thresholding and alert suppression, the global policy ceiling remained strictly negative (`HINDSIGHT_GRID_SCORE_POLICY_CEILING` = $-0.1983$, 95% CI: `[-0.2185, -0.1783]`). Although counterfactual patient-adaptive thresholding achieved positive utility ($+0.2819$, 95% CI: `[+0.2579, +0.3040]`), predictability modeling using admission and early-trajectory features yielded random-level discrimination (AUPRC = $0.2653$ vs. base rate $0.2608$), confirming that adaptive threshold needs are not reliably identifiable in advance (`REALISTIC_ACHIEVABLE_UTILITY` = $-0.1983$). A paired bootstrap comparison confirmed a statistically significant information/representation gap between perfect-information decision making and observable score policies ($\Delta = +1.0246$, 95% CI: `[+0.9997, +1.0494]`, $p < 0.0001$). Results were robust across $6$ distinct random initialization seeds (AUROC = $0.9609 \pm 0.0016$, Utility = $-0.2573 \pm 0.0020$).

**Conclusion:** High conventional discrimination (AUROC $0.9617$) did not translate into positive clinical utility under cross-hospital deployment. The observed utility deficit is consistent with a substantial information/representation gap between perfect-information decision making and observable scalar risk scores, compounded by false alarm accumulation in non-septic mimic hours. These findings demonstrate that evaluating predictive discrimination alone can mask severe operational failure modes, highlighting the need for decision-theoretic evaluation frameworks in clinical artificial intelligence.

---

## 1. INTRODUCTION

Early recognition of sepsis in intensive care units (ICUs) is a critical objective in clinical medicine. Sepsis, defined as life-threatening organ dysfunction caused by a dysregulated host response to infection, affects millions of hospitalized patients worldwide and remains a leading cause of hospital mortality (Singer et al., 2016). Because clinical outcomes deteriorate rapidly with delayed treatment, machine learning algorithms have been increasingly developed to forecast sepsis hours before clinical onset (Desautels et al., 2016; Nemati et al., 2018).

In clinical machine learning literature, model performance is predominantly benchmarked using conventional rank-ordering metrics, such as the Area Under the Receiver Operating Characteristic curve (AUROC) and the Area Under the Precision-Recall Curve (AUPRC). While these metrics measure a model's capacity to discriminate high-risk from low-risk observations across arbitrary decision boundaries, they fail to incorporate operational realities such as false alarm penalties, intervention lead times, and alert suppression constraints. In deployment, early-warning algorithms act as thresholded decision policies operating under asymmetric decision costs—where late alarms miss therapeutic windows, but repeated false alarms cause severe monitor alarm fatigue and workflow disruption (Cvach, 2012; Vickers & Elkin, 2006).

This disconnect becomes especially pronounced under cross-hospital deployment, where clinical data distributions shift across health system populations, electronic health record (EHR) systems, and practice patterns (Subbaswamy & Saria, 2020). Although recent deep learning models—including Time-Aware Transformers—demonstrate high discriminative ability on internal and external datasets, their true operational value under realistic clinical utility functions remains under-examined.

In this study, we investigate a central scientific question: **Why can a model with high conventional discrimination fail to produce positive net clinical utility under cross-hospital deployment?** 

Using the official PhysioNet/Computing in Cardiology Challenge 2019 benchmark (Reyna et al., 2019), we train a Time-Aware Transformer model ($M3$) on $20,336$ ICU stays from Emory University Hospital (Set A) and evaluate its deployable net utility on a held-out test cohort of $20,000$ ICU stays from Beth Israel Deaconess Medical Center (Set B). We formulate a **Dual-Bound Utility Decomposition Framework** that mathematically isolates the decision-theoretic potential of the action space (`GROUND_TRUTH_ORACLE_CEILING = +0.8262`), the diagnostic limits of observable score policies (`HINDSIGHT_GRID_SCORE_POLICY_CEILING = -0.1983`), the counterfactual headroom of patient-adaptive thresholding (`PATIENT_ADAPTIVE_THRESHOLD_CEILING = +0.2819`), and the deployable model utility (`FROZEN_MODEL_UTILITY = -0.2573`). Through rigorous patient-level bootstrap resampling ($B=1,000$) and multi-seed stability testing ($N=6$), we provide empirical proof that high AUROC ($0.9617$) masks an underlying information/representation gap, demonstrating why utility-centered evaluation is essential for clinical artificial intelligence.

---

## 2. RELATED WORK

### 2.1 Machine Learning for Early Sepsis Prediction
Machine learning models for sepsis early warning have evolved from simple physiological scoring systems—such as SIRS, SOFA, and qSOFA (Singer et al., 2016)—to complex supervised classifiers. Early data-driven approaches utilized logistic regression and tree-based ensembles on static or aggregated vital signs (Desautels et al., 2016). More recent studies have applied deep recurrent architectures and gradient boosting to continuous multivariate ICU time series (Nemati et al., 2018; Reyna et al., 2019).

### 2.2 Missingness-Aware Clinical Time-Series Modeling
Clinical ICU time series are characterized by extreme irregularity, variable sampling frequencies, and informative missingness patterns. Che et al. (2018) introduced GRU-D, demonstrating that explicitly modeling missingness masks and elapsed time deltas ($\Delta t$) allows recurrent networks to capture clinical sampling dynamics. Subsequent works confirmed that incorporating observation intervals substantially improves risk prediction in acute care data.

### 2.3 Transformer-Based Clinical Time-Series Models
Self-attention architectures have been adapted for clinical time series to capture long-range temporal dependencies without recurrent bottlenecking (Li et al., 2020). By integrating time-aware positional embeddings and missingness encodings, Time-Aware Transformers achieve state-of-the-art rank-ordering performance (AUROC $>0.95$) on competitive sepsis benchmarks.

### 2.4 Cross-Hospital Generalization and Domain Shift
Dataset shift represents a major barrier to deploying clinical machine learning models (Subbaswamy & Saria, 2020). Differences in patient demographics, disease prevalence, nursing measurement frequencies, and hospital coding practices frequently cause model performance to degrade when transferred to new medical centers. Domain adaptation methods, such as Domain-Adversarial Neural Networks (DANN; Ganin & Lempitsky, 2015), attempt to align feature representations across hospitals, but their impact on downstream clinical utility metrics remains poorly understood.

### 2.5 Decision-Theoretic Evaluation and Clinical Utility
Decision Curve Analysis (Vickers & Elkin, 2006) and utility-based scoring metrics (Reyna et al., 2019) move beyond AUROC by incorporating clinical costs and consequences. The PhysioNet 2019 Utility Metric explicitly penalizes false alarms ($-0.05$ pts/hr) and missed sepsis ($-2.0$ pts/patient) while rewarding timely early warnings ($+1.0$ pts). However, existing literature lacks formal decomposition frameworks to isolate whether utility failures stem from metric harshness, policy constraints, domain shift, or fundamental representation limits.

### 2.6 Positioning of This Study
Unlike prior work focusing solely on optimizing AUROC or proposing new neural architectures, this study evaluates a high-discrimination Time-Aware Transformer under formal cross-hospital deployment and introduces a dual-bound utility decomposition framework to rigorously diagnose utility breakdown.

---

## 3. DATASET AND PREPROCESSING

### 3.1 PhysioNet Challenge 2019 Dataset
We utilized the open-access PhysioNet/Computing in Cardiology Challenge 2019 dataset (Reyna et al., 2019), comprising $40,336$ adult ICU stays from two distinct hospital systems:
- **Set A (Emory University Hospital):** $20,336$ ICU stays ($1,790$ septic, $18,546$ non-septic).
- **Set B (Beth Israel Deaconess Medical Center):** $20,000$ ICU stays ($1,066$ septic, $18,934$ non-septic).

Each record consists of hourly observations containing $8$ vital sign variables, $26$ laboratory measurement variables, and $6$ demographic/admission variables.

### 3.2 Hospital Partitioning
To evaluate cross-hospital generalization without domain leakage, Set A (Emory) was designated exclusively for model training and validation, while Set B (BIDMC) was preserved as a completely held-out cross-hospital test cohort ($N=20,000$ patients, $753,927$ hourly observations).

### 3.3 Patient-Level Splitting
Set A was partitioned at the patient level into an in-domain training set ($16,192$ patients) and an in-domain validation set ($4,144$ patients) using a 80/20 stratified split based on sepsis prevalence. Patient overlap across all splits is zero ($0.0$).

### 3.4 Temporal Representation
Data were structured as hourly sequential vectors. For each hour $t$, observations contain vital signs (e.g., Heart Rate, $O_2$ Saturation, Temperature, Systolic Blood Pressure) and laboratory tests (e.g., WBC, Lactate, Platelets, Creatinine).

### 3.5 Missingness and Time-Delta Encoding
Following Che et al. (2018), missing raw values were forward-filled. For each feature $j \in \{1, \dots, 34\}$, we constructed:
1. Normalized raw value $v_j(t)$.
2. Binary missingness mask $m_j(t) \in \{0, 1\}$.
3. Elapsed time delta $\Delta t_j(t)$, representing hours since the last physical observation.

### 3.6 Label and Sepsis-Onset Definition
Clinical sepsis onset ($t_{\text{onset}}$) was defined per Sepsis-3 criteria (Singer et al., 2016), determined by the clinical concurrence of suspicion of infection and a sudden increase in SOFA score $\ge 2$ points. Hourly labels $y(t) = 1$ in the window $[t_{\text{onset}} - 6\text{h}, t_{\text{onset}} + 9\text{h}]$ and $0$ elsewhere.

### 3.7 Data Leakage Prevention
Feature normalization parameters (mean and standard deviation) were fit strictly on the in-domain training split ($16,192$ patients) and applied unchanged to validation and held-out test data. Decision thresholds were selected exclusively on validation predictions.

---

## 4. MODEL

### 4.1 M3 Time-Aware Transformer
The $M3$ model is a Time-Aware Transformer (`TACTModel`) engineered for irregular clinical time series.

### 4.2 Input Representation
At each hour $t$, the input vector $\mathbf{x}(t) \in \mathbb{R}^{102}$ concatenates normalized values $\mathbf{v}(t) \in \mathbb{R}^{34}$, missingness masks $\mathbf{m}(t) \in \mathbb{R}^{34}$, and time deltas $\mathbf{\Delta t}(t) \in \mathbb{R}^{34}$.

### 4.3 Temporal Encoding
Inputs are projected through a linear embedding layer to dimension $d_{\text{model}} = 64$ and combined with sinusoidal positional encodings to preserve sequence order.

### 4.4 Transformer Encoder
The model features $3$ Transformer encoder layers, each incorporating $4$ multi-head self-attention heads, layer normalization, dropout ($p=0.10$), and feed-forward sub-layers with GELU activations.

### 4.5 Prediction Head
Hourly hidden states $\mathbf{h}(t) \in \mathbb{R}^{64}$ pass through a linear layer to generate uncalibrated logits $z(t)$, which are transformed via sigmoid activation to yield continuous risk probabilities $p(t) \in (0, 1)$.

### 4.6 Training Objective
The network was trained using Binary Cross-Entropy with Logits loss, incorporating a positive class weight ($w_{\text{pos}} \approx 11.2$) to compensate for hourly label imbalance. Optimization utilized AdamW ($\text{lr} = 10^{-4}, \text{weight\_decay} = 10^{-4}$) with ReduceLROnPlateau scheduling for up to $30$ epochs.

---

## 5. EVALUATION FRAMEWORK

### 5.1 Conventional Discrimination Metrics
Model discrimination was evaluated using AUROC, AUPRC, Brier Score, and Expected Calibration Error (ECE) across all $753,927$ hourly records in the held-out BIDMC test set.

### 5.2 PhysioNet Utility Function
Net clinical utility was computed using the official PhysioNet 2019 metric scoring function $U(S, Y)$:
- **Optimal TP Credit (+1.0):** Single alarm at $t_{\text{optimal}} = \max(0, t_{\text{onset}} - 6\text{h})$.
- **Early/Late Penalty:** Linear ramp from $0.0$ at $t_{\text{early}} = 12\text{h}$ pre-onset to $+1.0$ at $t_{\text{optimal}}$, decaying to $0.0$ at $t_{\text{late}} = 3\text{h}$ post-onset.
- **False Alarm Penalty:** $-0.05$ points per hour.
- **Missed Sepsis Penalty:** $-2.0$ points per patient.
- **Normalization:** Total achieved points divided by total maximum possible points ($N_{\text{sepsis}} \times 1.0 = 1,066.0$).

### 5.3 Deployable Policy
The deployable policy converts probabilities $p(t)$ to binary alarms using a prespecified validation-selected threshold ($th_{\text{val}}^* = 0.190$) and a $36$-hour alert suppression cooldown ($C = 36\text{h}$).

### 5.4 Ground-Truth Oracle Ceiling
The `GROUND_TRUTH_ORACLE_CEILING` evaluates maximum achievable utility under perfect information using true sepsis labels $y_{\text{true}}$ and onset times $t_{\text{onset}}$ only. It uses **zero** model scores, probabilities, logits, or predictions.

### 5.5 Hindsight Score-Policy Ceiling
The `HINDSIGHT_GRID_SCORE_POLICY_CEILING` sweeps thresholds $th \in [0.005, 0.995]$ and cooldowns $C \in \{6, \dots, 336\}\text{h}$ on held-out test predictions in hindsight to find the peak achievable score policy utility.

### 5.6 Patient-Adaptive Ceiling
The `PATIENT_ADAPTIVE_THRESHOLD_CEILING` evaluates a counterfactual diagnostic bound where each patient's threshold $th_i^*$ is selected in hindsight to maximize that individual patient's utility.

### 5.7 Utility Gap Decomposition
We define the Information/Representation Gap as:
$$\Delta_{\text{Info}} = \text{GROUND\_TRUTH\_ORACLE\_CEILING} - \text{HINDSIGHT\_GRID\_SCORE\_POLICY\_CEILING}$$

### 5.8 Bootstrap Uncertainty
Uncertainty was quantified using patient-level bootstrap resampling ($B = 1,000$ iterations), resampling whole patients with replacement. 95% Confidence Intervals were computed via the $2.5^{\text{th}}$ and $97.5^{\text{th}}$ percentiles.

### 5.9 Multi-Seed Stability
Model stability was evaluated across $6$ distinct random initialization seeds (Seed $42$ original + Seeds $1, 2, 3, 4, 5$), training each model from scratch.

### 5.10 Predictability Analysis
Logistic Regression and Gradient Boosted Trees were trained to predict patient-adaptive threshold needs using admission-time ($t=0$) and early-trajectory ($t \in [0, 5]$) features.

---

## 6. EXPERIMENTAL RESULTS

### 6.1 Conventional Model Discrimination
On the held-out BIDMC test set ($N=20,000$), the frozen $M3$ Transformer achieved excellent conventional discrimination:
- **AUROC:** `0.961726` (`0.9617`)
- **AUPRC:** `0.423114` (`0.4231`)
- **Brier Score:** `0.015290`
- **ECE:** `0.018151`

### 6.2 Cross-Hospital Utility
Despite high discrimination, deployable net utility under prespecified validation parameters ($th=0.190, C=36\text{h}$) was strictly negative:
$$\text{FROZEN\_MODEL\_UTILITY} = \mathbf{-0.257312} \quad (95\%\text{ CI: }[-0.282823, -0.233519])$$

### 6.3 Oracle Decomposition
Independent re-computations confirmed the dual-bound utility decomposition with $0.000000\text{e}+00$ discrepancy ($\le 10^{-10}$):
- `GROUND_TRUTH_ORACLE_CEILING`: **`+0.826246`** ($+0.826245570148$)
- `PATIENT_ADAPTIVE_THRESHOLD_CEILING`: **`+0.281895`**
- `HINDSIGHT_GRID_SCORE_POLICY_CEILING`: **`-0.198307`** (at $th=0.345, C=72\text{h}$)
- `FROZEN_MODEL_UTILITY`: **`-0.257312`**
- `RAW_SCORE_POLICY_CEILING`: **`-0.855545`** (at $th=0.745, C=0\text{h}$)

### 6.4 Bootstrap Confidence Intervals
Patient-level bootstrap resampling ($B = 1,000$) demonstrated that **zero 95% Confidence Intervals cross 0.0**:
- `FROZEN_MODEL_UTILITY`: `[-0.282823, -0.233519]` (Strictly Negative)
- `HINDSIGHT_GRID_SCORE_POLICY_CEILING`: `[-0.218529, -0.178330]` (Strictly Negative)
- `GROUND_TRUTH_ORACLE_CEILING`: `[+0.806653, +0.844781]` (Strictly Positive)

Paired significance tests confirmed that the Information/Representation Gap is statistically significant:
$$\Delta_{\text{Info}} = +1.024585 \quad (95\%\text{ CI: }[+0.999690, +1.049449], \, p < 0.0001)$$

### 6.5 Multi-Seed Stability
Training $M3$ across $6$ distinct seeds produced tightly clustered results:
- **AUROC:** $0.9609 \pm 0.0016$ (range: $0.9584$ to $0.9631$)
- **AUPRC:** $0.4224 \pm 0.0026$ (range: $0.4189$ to $0.4265$)
- **`FROZEN_MODEL_UTILITY`:** $-0.257316 \pm 0.002012$ (range: $-0.259850$ to $-0.254128$)
- **`HINDSIGHT_GRID_CEILING`:** $-0.198802 \pm 0.001736$ (range: $-0.201210$ to $-0.196144$)

Not a single seed achieved positive utility, confirming that the negative result is not an artifact of random initialization.

### 6.6 Patient-Adaptive Ceiling
Counterfactual patient-adaptive thresholding achieved positive utility (`+0.281895`). A total of $278$ of $1,066$ septic patients ($26.08\%$) achieved higher utility under patient-specific thresholds.

### 6.7 Predictability of the Adaptive Ceiling
Predictability modeling to identify patients needing adaptive thresholds achieved a test AUPRC of $0.2653$ (base rate $0.2608$), which is virtually identical to random guessing. Thus, adaptive threshold needs are not predictable in advance, keeping `REALISTIC_ACHIEVABLE_UTILITY` strictly negative (**`-0.198307`**).

### 6.8 Error/Utility Analysis
False alarms on non-septic patients (penalized at $-0.05$ pts/hr) accumulated $-2,146.75$ penalty points across $18,934$ non-septic patients, overwhelming the $+880.78$ points earned on septic patients.

---

## 7. DISCUSSION

### 7.1 The Discrimination–Utility Disconnect
Our findings illustrate a striking operational paradox: a model achieving high conventional discrimination (AUROC $0.9617$) fails completely under deployment ($U = -0.2573$). This disconnect occurs because AUROC evaluates rank-ordering across all hourly records regardless of temporal sequencing or asymmetric penalties.

### 7.2 Why High AUROC Did Not Translate to Positive Utility
In ICU monitoring, non-septic patients contribute $>98\%$ of all hourly data ($726,927$ out of $753,927$ hours). Even at $99.34\%$ hourly specificity, false alarms trigger on $0.66\%$ of non-septic hours. Under an asymmetric penalty structure ($-0.05$ pts/hr vs. $+1.0$ max TP credit), non-septic false alarm accumulation outweighs positive true positive credit.

### 7.3 Information/Representation Gap
The statistically significant separation between `GROUND_TRUTH_ORACLE_CEILING` ($+0.8262$) and `HINDSIGHT_GRID_SCORE_POLICY_CEILING` ($-0.1983$) ($\Delta = +1.0246, p < 0.0001$) indicates that scalar probability outputs $p(t)$ derived from observable clinical features do not preserve sufficient temporal risk information to support positive utility under a global threshold policy.

### 7.4 Policy Optimization vs. Representation Improvement
Dense 2D grid sweeps across thresholds ($th \in [0.005, 0.995]$) and alert suppression cooldowns ($C \in [6, 336]\text{h}$) demonstrated an interior peak at $C=72\text{h}$ ($U=-0.1983$), after which utility turned over and decreased. This proves that global policy optimization cannot overcome the utility deficit.

### 7.5 Clinical Interpretation
From a clinical perspective, non-septic ICU patients presenting with Systemic Inflammatory Response Syndrome (SIRS) or fever generate physiological trajectories that mimic early sepsis. Scalar risk models assign elevated risk scores to these mimic patients, triggering unavoidable false alarm burdens.

### 7.6 Implications for Evaluation of Clinical AI
These results demonstrate that benchmarking clinical AI models on AUROC/AUPRC alone creates an incomplete—and potentially misleading—picture of deployment readiness. Future clinical AI evaluations must integrate utility metrics, action-space constraints, and decision-ceiling decompositions.

---

## 8. LIMITATIONS

1. **Retrospective Evaluation:** Analysis was conducted on retrospective observational ICU data (PhysioNet 2019 dataset).
2. **Two-Hospital Setting:** Cross-hospital transfer was evaluated between two major academic medical centers (Emory and BIDMC).
3. **Utility Function Sensitivity:** Results depend on the specific parameterization of the PhysioNet 2019 Utility Metric.
4. **Counterfactual Headroom:** The patient-adaptive threshold ceiling (`+0.281895`) is a hindsight diagnostic quantity and is not deployable.
5. **Feature Scope:** Predictability modeling evaluated specific admission ($t=0$) and early-trajectory ($t \in [0, 5]$) feature sets; richer feature spaces may yield different predictability.
6. **Lack of Prospective Validation:** Prospective workflow integration and clinician alarm response were not evaluated.

---

## 9. CONCLUSION

This study demonstrates that high predictive discrimination (AUROC $0.9617$) does not guarantee positive net clinical utility under cross-hospital deployment. Through a formal Dual-Bound Utility Decomposition Framework, we proved that positive utility is mathematically achievable on BIDMC under the official action space (`+0.8262`), but deployable model performance remains strictly negative (`-0.2573`). This failure is consistent with a substantial information/representation gap between perfect-information decision making and observable scalar risk scores, compounded by false alarm accumulation in non-septic mimic hours. Our findings highlight the necessity of decision-theoretic evaluation frameworks that explicitly separate discriminative ranking from deployable clinical utility.

---

## REFERENCES

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
