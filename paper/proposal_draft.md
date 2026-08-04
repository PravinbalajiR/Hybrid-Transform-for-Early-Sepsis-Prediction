# Proposal & System Design Document

# A Knowledge-Guided Hybrid Transformer for Early Sepsis Prediction: Organ-Aware Representations and Missingness Encoding for Irregular ICU Time-Series

---

## 1. Abstract

Early prediction of sepsis in Intensive Care Units (ICUs) is a critical clinical challenge where every hour of delayed detection increases patient mortality by 7.6%. Existing machine learning approaches on public ICU benchmarks—such as the PhysioNet/CinC 2019 Challenge dataset—suffer from two major limitations: (1) treating all 34 physiological variables as an undifferentiated feature vector in standard temporal architectures (LSTM/GRU/Transformer), ignoring domain-specific organ system hierarchy, and (2) treating missing values either through ad-hoc imputation (destroying informative observation patterns) or attempting handcrafted clinical scoring systems (SOFA/qSOFA) that cannot be fully computed due to dataset constraints (e.g., absence of Glasgow Coma Scale and vasopressor dosages).

To address these gaps, we propose a **Knowledge-Guided Dual-Branch Hybrid Transformer**. Our framework combines:
1. An **Organ-Knowledge Branch** that groups physiological variables into 6 clinically distinct organ systems (Cardiovascular, Respiratory, Renal, Liver, Metabolic/Haematological, Temperature) processed via specialized sub-encoders;
2. A **Data-Driven Time-Aware Transformer Branch** that embeds raw continuous values alongside explicit observation masks ($M_{t,j}$) and time-delta intervals ($\Delta_{t,j}$); and
3. A **Cross-Attention Fusion Layer** that dynamically aligns organ-system dysfunction representations with temporal trajectory embeddings.

Evaluated on 40,336 ICU patients using the official PhysioNet 2019 Utility Score under a strict cross-hospital deployment split (Hospital A for training/validation, Hospital B for testing), this architecture aims to establish a new state-of-the-art while maintaining clinical interpretability.

---

## 2. Problem Formulation & Dataset Constraints

### 2.1 Problem Definition
Given a multivariate, irregularly sampled time-series $X_{1:T} = (\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T)$ recorded up to ICU hour $T$, where each hour contains $D=34$ physiological and laboratory variables plus $D_{demo}=5$ static demographic attributes, the goal is to predict the binary label $y_{T+\tau} \in \{0, 1\}$ indicating whether sepsis onset occurs within a prediction window of $\tau = 6$ hours.

### 2.2 Dataset Schema & Key Limitations
The PhysioNet/CinC 2019 Challenge dataset contains 40,336 ICU patients across two datasets:
- **Set A**: 20,336 patients
- **Set B**: 20,000 patients

We adopt a strict **Set A to Set B validation split**: training and hyperparameter validation are conducted exclusively on Set A, while Set B serves as a completely held-out evaluation set to assess cross-site model generalization.

#### Confirmed Schema Limitations
Direct audit of the raw `.psv` schema confirms the following missing variables required for full clinical SOFA (Sequential Organ Failure Assessment) computation:
- **Glasgow Coma Scale (GCS)**: Absent.
- **Vasopressor Dosing Rates**: Absent.
- **PaO2**: Absent (only SaO2 and FiO2 available).

Consequently, any baseline claiming to compute exact clinical SOFA scores on this specific dataset is methodologically flawed. Our hybrid architecture explicitly overcomes this by learning data-driven organ dysfunction embeddings rather than relying on incomplete clinical formulas.

---

## 3. Dataset Feasibility Audit (Empirical Results)

An exhaustive feasibility audit across all 40,336 patient files yielded the following empirical findings.

> **Methodological Note on Missingness Calculation**: All missingness percentages reported below are computed **across all patient-hour records** (i.e., the total fraction of hourly observation slots where the variable was unobserved), rather than the fraction of patients who never had the lab drawn.

### 3.1 Class Balance & ICU Stay Dynamics
- **Total Patients**: 40,336
- **Sepsis Positive**: 2,932 patients (**7.27%**)
- **Non-Sepsis**: 37,404 patients (**92.73%**)
- **Class Ratio**: $1 : 12.8$ (severe class imbalance requiring focal/weighted loss formulation)
- **ICU Stay Duration**:
  - Non-sepsis cohort: Mean = 36.9 hours (Median = 39 hours, Max = 336 hours)
  - Sepsis cohort: Mean = 58.8 hours (Median = 38 hours, Max = 336 hours)

### 3.2 Patient-Hour Missingness Hierarchy
Empirical missingness rates across all 34 hourly features (evaluated across all patient-hours) reveal a clear bimodal distribution:

```
[Always-On Vitals]               [Routine Labs]               [Rare Markers]
HR (9.9%), MAP (12.5%)  ───►  Glucose (82.9%), WBC (93.6%) ───► Bilirubin_direct (99.8%)
SBP (14.6%), O2Sat (13.1%)     Creatinine (93.9%), BUN (93.1%)   Fibrinogen (99.3%)
Resp (15.4%), DBP (31.3%)      Lactate (97.3%), ABGs (93-96%)    TroponinI (99.0%)
```

#### Summary by Organ System:
1. **Liver System**: **98.8%** mean missingness (*AST, Alkalinephos, Bilirubin_direct, Bilirubin_total*)
2. **Renal System**: **93.9%** mean missingness (*Creatinine, BUN, Chloride, Calcium, Potassium, Magnesium, Phosphate*)
3. **Metabolic & Haematological**: **93.8%** mean missingness (*Glucose, Lactate, BaseExcess, HCO3, WBC, Hct, Hgb, PTT, Fibrinogen, Platelets*)
4. **Respiratory System**: **71.5%** mean missingness (*O2Sat, Resp, EtCO2, FiO2, PaCO2, SaO2, pH*)
5. **Temperature**: **66.2%** missingness (*Temp*)
6. **Cardiovascular System**: **33.5%** mean missingness (*HR, SBP, MAP, DBP, TroponinI*)

---

## 4. Proposed Architecture Design

```
                          PhysioNet 2019 Hourly Inputs
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
  [Knowledge Branch: Organ-Aware]                        [Data-Driven Branch: Temporal]
  ───────────────────────────────                        ────────────────────────────
  Organ Partitioning:                                    Triple Input Vector per feature j:
   • Cardio: {HR, SBP, MAP, DBP, TropI}                   • Scaled Value: x̂_{t,j}
   • Resp:   {O2Sat, Resp, FiO2, PaCO2...}                • Binary Mask:  m_{t,j} ∈ {0,1}
   • Renal:  {Creatinine, BUN, K+, Ca2+...}               • Time Delta:   Δ_{t,j} (hrs since last obs)
   • Liver:  {AST, AlkPhos, Bilirubin...}
   • Met/Hem:{Glucose, Lactate, WBC...}                  Concatenation: z_{t} ∈ ℝ^{3F}
   • Temp:   {Temp} + Shock Index                        Linear Project + Positional Encoding
            │                                                     │
   Per-Organ Encoder (Small MLPs/GRUs)                   Time-Aware Transformer Encoder
   Produces Organ Vectors e_{t,k} ∈ ℝ^{d_o}              Produces Sequence Embeddings h_t ∈ ℝ^{d_m}
            │                                                     │
            └──────────────────────────┬──────────────────────────┘
                                       │
                            Cross-Attention Fusion Layer
                            Query:  Transformer Temporal Context h_t
                            Keys/Values: Organ Group Embeddings {e_{t,1}, ..., e_{t,6}}
                                       │
                            Unified Latent Vector u_t
                                       │
                  ┌────────────────────┴────────────────────┐
                  ▼                                         ▼
         Sepsis Prediction Head               Auxiliary Organ Dysfunction Task
         P(Sepsis within 6h) ∈ [0, 1]         Predict multi-organ collapse score
                  │                                         │
                  └────────────────────┬────────────────────┘
                                       │
                           MC Dropout Uncertainty Layer
                           (Confidence-gated clinical alerts)
```

### 4.1 Knowledge Branch: Organ Partitioning
The input vector $\mathbf{x}_t \in \mathbb{R}^{34}$ is explicitly split into 6 subsets $S_k \subset \{1, \dots, 34\}$ representing physiological organ domains:
- $\mathbf{e}_{t, \text{cardio}} = \text{MLP}_{\text{cardio}}([ \mathbf{x}_{t, \text{cardio}}, \mathbf{m}_{t, \text{cardio}}, \Delta_{t, \text{cardio}} ])$
- $\mathbf{e}_{t, \text{resp}} = \text{MLP}_{\text{resp}}([ \mathbf{x}_{t, \text{resp}}, \mathbf{m}_{t, \text{resp}}, \Delta_{t, \text{resp}} ])$
- $\mathbf{e}_{t, \text{renal}} = \text{MLP}_{\text{renal}}([ \mathbf{x}_{t, \text{renal}}, \mathbf{m}_{t, \text{renal}}, \Delta_{t, \text{renal}} ])$
- $\mathbf{e}_{t, \text{liver}} = \text{MLP}_{\text{liver}}([ \mathbf{x}_{t, \text{liver}}, \mathbf{m}_{t, \text{liver}}, \Delta_{t, \text{liver}} ])$
- $\mathbf{e}_{t, \text{metab}} = \text{MLP}_{\text{metab}}([ \mathbf{x}_{t, \text{metab}}, \mathbf{m}_{t, \text{metab}}, \Delta_{t, \text{metab}} ])$
- $\mathbf{e}_{t, \text{temp}} = \text{MLP}_{\text{temp}}([ \mathbf{x}_{t, \text{temp}}, \mathbf{m}_{t, \text{temp}}, \Delta_{t, \text{temp}} ])$

Each organ encoder outputs a fixed $d_o$-dimensional organ state embedding.

### 4.2 Data-Driven Branch: Time-Aware Transformer
For every feature $j \in \{1, \dots, F\}$, we construct a triplet $(x_{t,j}, m_{t,j}, \Delta_{t,j})$:
$$\mathbf{z}_t = [\mathbf{x}_t \odot \mathbf{m}_t \,\|\, \mathbf{m}_t \,\|\, \Delta_t] \in \mathbb{R}^{3F}$$
The sequence $(\mathbf{z}_1, \dots, \mathbf{z}_T)$ is projected to model dimension $d_m$, added to sinusoidal positional encodings, and passed through a $L$-layer Multi-Head Self-Attention Transformer Encoder.

### 4.3 Cross-Attention Fusion
The temporal representation $\mathbf{h}_t \in \mathbb{R}^{d_m}$ acts as the Query $Q$, while the concatenated organ embeddings $E_t = [\mathbf{e}_{t,1}, \dots, \mathbf{e}_{t,6}] \in \mathbb{R}^{6 \times d_o}$ serve as Keys $K$ and Values $V$:
$$\mathbf{u}_t = \text{MultiHeadAttention}(Q=\mathbf{h}_t W_Q, K=E_t W_K, V=E_t W_V)$$

---

## 5. Ablation Suite & Baseline Comparison

To rigorously validate each component, the evaluation suite benchmarks 5 distinct model variants on identical splits:

| Model Index | Variant | Organ Branch | Transformer | Missingness Encoding ($\mathbf{m}, \Delta$) |
|:---:|---|:---:|:---:|:---:|
| **M1** | XGBoost Baseline | ❌ | ❌ | ❌ (Mean/LOCF Imputed) |
| **M2** | Plain Transformer | ❌ | ✅ | ❌ (Naive Mean Imputed) |
| **M3** | Time-Aware Transformer | ❌ | ✅ | ✅ (Triplet Vector) |
| **M4** | Organ Branch Only | ✅ | ❌ | ✅ |
| **M5** | **Proposed Hybrid Transformer** | ✅ | ✅ | ✅ |

### Benchmark Ceiling
Published PhysioNet 2019 Challenge winning utility scores (~0.36–0.43 Utility) serve as the external performance benchmark.

---

## 6. Evaluation Protocols & Metrics

1. **Primary Metric**: Official PhysioNet 2019 Utility Score $U_{\text{total}}$ (rewarding early detection up to 6h before onset, penalizing false alarms at $-0.05/\text{hr}$ and missed sepsis at $-2.0$).
2. **Secondary Metrics**: AUROC, AUPRC, F1-Score, Sensitivity, Specificity.
3. **Timing Analysis**: Mean lead time (hours prior to clinical onset) for true positive detections.
4. **Calibration & Uncertainty**: Expected Calibration Error (ECE) and MC Dropout variance confidence intervals.

---

## 7. 16-Week Implementation Roadmap

- **Week 1 (Current)**: Feasibility audit, missingness visualization, literature table, baseline setup. [COMPLETED]
- **Week 2**: Proposal document locking, exploratory data analysis notebook assembly.
- **Week 3–4**: Data preprocessing pipeline (masks, deltas, Z-score normalizer, hospital-stratified splits).
- **Week 4–5**: Organ feature engineering & Shock Index construction.
- **Week 5–6**: Baseline models (XGBoost, Plain Transformer) training & Utility logging.
- **Week 6–7**: **MVP Build**: Single-pathway Transformer with organ-grouped inputs.
- **Week 7–8**: Integrate observation mask & time-delta encoding into MVP (Defensible Fallback Result).
- **Week 8–9**: Architecture split: Organ-Knowledge Branch + Temporal Transformer Branch.
- **Week 9–10**: Cross-Attention Fusion Layer implementation & tuning.
- **Week 10–11**: Multi-task head (sepsis prediction + organ dysfunction embedding).
- **Week 11–12**: Full ablation suite execution (M1–M5).
- **Week 12–13**: Uncertainty module (MC Dropout) & ECE calibration analysis.
- **Week 13–14**: Patient case studies & interpretability analysis.
- **Week 14–16**: Manuscript assembly, figure polish, & target journal submission.
