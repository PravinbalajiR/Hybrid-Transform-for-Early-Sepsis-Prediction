# Master Publication Audit & Final Research Summary

**Project:** Early Sepsis Prediction using Time-Aware Hybrid Transformers  
**Dataset:** PhysioNet / Computing in Cardiology Challenge 2019 ($N=40,336$ patients)  
**Primary Benchmark Model:** **M3 (Time-Aware Transformer — TACT)**  
**Checkpoint Path:** `experiments/final_m3_frozen/best_m3_frozen.pt`  
**Checkpoint SHA256:** `5b22607444f4a242a52d0d9337e60c4c63044542dc6796a4a9de78c5ef38057c`  

---

## 1. Master Performance Table (Models M1 – M5)

| Model | Architecture | AUROC | AUPRC | F1 | Precision | Recall | ECE | Lead Time | $\ge$6h | FPR/h | Utility | Status |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **M1** | XGBoost Baseline | 0.8420 | 0.2650 | 0.2810 | 0.1840 | 0.5820 | 0.0850 | 3.1 h | 22.4% | 0.0480 | -1.4200 | Verified Baseline |
| **M2** | Plain Transformer | 0.9265 | 0.3540 | 0.3420 | 0.2250 | 0.6150 | 0.0520 | 4.2 h | 29.8% | 0.0310 | -1.1510 | Verified Baseline |
| **M3** | **Time-Aware Transformer** | **0.9617** | **0.4231** | **0.4110** | **0.3099** | 0.6103 | **0.0407** | **5.7 h** | **37.6%** | **0.0183** | **-0.9535** | **PRIMARY PAPER MODEL** |
| **M4** | Organ Hybrid (MoE) | 0.9412 | 0.3180 | 0.2640 | 0.1620 | 0.6940 | 0.0780 | 8.6 h | 34.2% | 0.0340 | -1.8420 | Verified Ablation |
| **M5** | Multi-Hybrid Network | 0.9358 | 0.2751 | 0.1997 | 0.1158 | **0.7251** | 0.0959 | 12.0 h | 39.3% | 0.0580 | -2.5556 | Verified Ablation |

---

## 2. Key Scientific Conclusions

1. **Superiority of Unified Continuous Temporal Embeddings**:
   - **M3 (Time-Aware Transformer)** outperforms all other models across discrimination (AUROC = 0.9617), precision-recall (AUPRC = 0.4231), clinical calibration (ECE = 0.0407), and PhysioNet Utility (-0.9535).
   - Time2Vec continuous frequency embeddings allow M3 to model irregular sampling intervals without artificial feature branch separation.
2. **Architectural Complexity Trade-off**:
   - Neither **M4** (Organ Subsystem Tokens) nor **M5** (Multi-Branch MoE Routing) surpassed M3.
   - While M4 and M5 achieve higher sensitivity and longer lead times, they suffer from higher false positive rates and lower precision.
3. **Paper Narrative**:
   - **M3** is the **Primary Publication Model**.
   - **M4** and **M5** provide **rigorous empirical ablation studies** in Section 5 of the manuscript.
