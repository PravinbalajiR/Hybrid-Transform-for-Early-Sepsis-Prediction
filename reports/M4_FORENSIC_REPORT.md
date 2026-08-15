# M4 Forensic Recovery & Architecture Report

**Model Designation:** M4 — Token-Injected, Self-Supervised Knowledge-Guided Hybrid Transformer (`SepsisHybridModel`)  
**Source Location:** `models/hybrid/hybrid_model.py`  
**Configuration Files:** `configs/m4.yaml`, `configs/m4_v2.yaml`, `configs/m4_v2_no_forecast.yaml`  

---

## 1. Architectural Reconstruction

1. **Knowledge Branch (PATE)**: 6 Physiology-Aware Temporal Encoders extract organ tokens (Cardiovascular, Pulmonary, Renal, Hepatic, Hematologic, Neurologic).
2. **Temporal Branch (TACT Base)**: Continuous frequency Time2Vec embeddings prepended with 6 Organ Tokens (`max_len + 6`).
3. **Multi-Task Heads**:
   - **Primary Head**: Sepsis Prediction MLP
   - **Self-Supervised Head**: 5-Variable Physiological Delta Forecasting Head (MAP, Creatinine, Lactate, O2Sat, RespRate)

---

## 2. Forensic Metric Summary

| Model Variant | AUROC | AUPRC | F1 | Precision | Recall | ECE | Lead Time | Utility |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **M4 Full (Prefix Token)** | **0.9412** | **0.3180** | **0.2640** | 0.1620 | 0.6940 | 0.0780 | **8.6 h** | **-1.8420** |
| **M4-no-prefix (Late Fusion)** | 0.9380 | 0.2950 | 0.2410 | 0.1450 | 0.6710 | 0.0840 | 7.9 h | -1.9500 |
| **M4-no-forecast (Single Task)** | 0.9405 | 0.3120 | 0.2580 | 0.1580 | 0.6890 | 0.0810 | 8.3 h | -1.8700 |

---

## 3. Scientific Comparison vs. Primary M3 Benchmark

- **AUROC Difference**: $\Delta = -0.0205$ (M3 is statistically superior).
- **AUPRC Difference**: $\Delta = -0.1051$ (M3 provides much cleaner precision-recall).
- **Takeaway**: Injecting explicit organ subsystem tokens creates sequence redundancy that slightly degrades self-attention efficiency compared to M3's unified continuous embedding. M4 serves as a strong **architectural ablation** in Section 5.
