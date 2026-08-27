# FULL CODE AND SCIENTIFIC RECTIFICATION AUDIT

**Repository:** `Hybrid-Transform-for-Early-Sepsis-Prediction`  
**Branch:** `paper-v1.0`  
**Baseline Commit:** `0d2acdbb36dfecab7d6f51cbffeb3ea0cf07c4bc` (`0d2acdb`)  
**Baseline Tag:** `paper-v1.0-baseline-frozen`

---

## 1. RECTIFICATION AUDIT TABLE

| Issue # | Component / File | Current Implementation | Scientific Risk | Required Correction | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **1** | `models/transformer/tact_model.py` | `nn.TransformerEncoder` called without `mask` or `is_causal=True`. | **Causality Leakage:** Position $t$ attends to future time steps $t' > t$, leaking future ICU measurements into hourly predictions. | Implement explicit upper-triangular causal attention mask (`torch.triu(..., diagonal=1)`) so $t$ depends strictly on $t' \le t$. | **Identified** |
| **2** | `models/organ_branch/organ_encoders.py` | `shock_index = hr - sbp` | **Incorrect Physiology Formula:** Subtraction `HR - SBP` is dimensionally invalid and mathematically incorrect. | Correct definition to `Shock Index = HR / (SBP + 1e-5)` with safe clipping for zero/near-zero SBP. Centralize in preprocessing. | **Identified** |
| **3** | `configs/feature_schema.yaml` | Hardcoded dimensions (`34`, `68`, `102`) scattered across models and scripts. | **Brittle Architecture & Schema Mismatch:** Modifying feature inputs breaks hardcoded tensor assertions. | Create a single canonical `configs/feature_schema.yaml` defining feature ordering, categories, and derived schema dimensions. | **Identified** |
| **4** | Preprocessing & Derived Features | Derived features calculated ad-hoc inside specific model sub-modules. | **Inconsistent Feature Representation:** Different models calculate or omit derived physiological features differently. | Single source-of-truth preprocessing pipeline producing all derived features, baseline normalization, and dynamics. | **Identified** |
| **5** | Missingness & Temporal Representation | `value + mask + delta` ($3 \times 34 = 102$) without decay/reliability model. | **Information Decay Unrepresented:** Stale observations retain static weight regardless of elapsed time delta $\Delta t$. | Add explicit mathematical Temporal Reliability layer $R(t) = \exp(-\gamma \cdot \Delta t)$ to weight historical state confidence. | **Identified** |
| **6** | Patient-Adaptive Physiological Baselines | Standard population zero-mean standardization only. | **Patient Variability Ignored:** Normal baseline vitals vary widely across individuals (e.g. baseline HR 55 vs 95). | Implement causal patient-adaptive baseline deviation $\mathbf{v}_{\text{dev}}(t) = \mathbf{v}(t) - \mathbf{\mu}_{\text{patient}}(\le t)$ using only $t' \le t$. | **Identified** |
| **7** | Physiological Deterioration Dynamics | Static physiological level $\mathbf{v}(t)$ only. | **Trajectory Unrepresented:** Ignores physiological rate of change (e.g., rapid MAP drop vs stable low MAP). | Implement causal velocity $\mathbf{v}'(t) = \frac{\Delta \mathbf{v}}{\Delta t}$ and acceleration $\mathbf{v}''(t) = \frac{\Delta \mathbf{v}'}{\Delta t}$. | **Identified** |
| **8** | Early-Warning Horizon & Objectives | Single binary 6-hour prediction head with standard cross-entropy loss. | **Single-Horizon Bottleneck:** Does not distinguish early warning lead times (6h, 12h, 24h) or penalize late alarms. | Implement Multi-Horizon prediction head (6h, 12h, 24h) with lead-time-aware training objective. | **Identified** |
| **9** | Robustness & Missing Sensor Stress Test | Evaluated only under natural observation missingness. | **Uncertain Sensor Resilience:** Vulnerable to complete sensor failures or missing clinical variable groups in real ICUs. | Add systematic 0%–50% sensor dropout experiments to evaluate degradation under missing variable groups. | **Identified** |
| **10** | Validation & Verification Gate | No automated future-information leakage test. | **Unverified Causality Claims:** Risk of silent temporal leakage in novel layers. | Create `scripts/test_future_information_invariance.py` to rigorously verify zero future-information sensitivity. | **Identified** |

---

## 2. AUDIT VERDICT AND PLAN

Phase A complete. All 10 technical issues identified and mapped to required corrections. Proceeding to Phase B (Causal Temporal Modeling & Future Invariance Testing).
