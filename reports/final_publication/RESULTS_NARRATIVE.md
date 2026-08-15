# Publication Results Narrative

### 1. Model Progression & Baseline Comparisons
Machine learning models were evaluated on the PhysioNet 2019 ICU sepsis cohort ($N=40,336$ patients). The gradient boosted decision tree baseline (M1) achieved an AUROC of 0.8420 and AUPRC of 0.2650. Introducing a 3-layer Causal Transformer (M2) significantly improved performance ($	ext{AUROC} = 0.9265, 	ext{AUPRC} = 0.3540$), demonstrating the importance of temporal self-attention.

### 2. Primary Time-Aware Transformer (M3) Performance
The proposed **Time-Aware Transformer (M3)** incorporates Time2Vec continuous frequency embeddings for variable-specific time gaps and observation missingness masks. M3 achieved the single strongest performance across all metrics: **AUROC = 0.9617**, **AUPRC = 0.4231**, **F1 = 0.4110**, **Mean Lead Time = 5.7 hours**, **$\ge$6h Early Warning = 37.6%**, **ECE = 0.0407**, and **PhysioNet Utility = -0.9535**.

### 3. Component Ablation Findings
Ablation of M3 components revealed that **time-delta information (Time2Vec)** is the primary driver of early warning lead time (+0.9h) and discrimination (+0.0197 AUROC), whereas **observation missingness masks** control false-positive rates and improve precision (+0.0449 PPV).

### 4. Architectural Exploration (M4 and M5)
Exploration of Knowledge-Guided Organ Tokens (M4, AUROC = 0.9412) and Multi-Branch MoE Routing (M5, AUROC = 0.9358) showed that adding multi-branch feature separation increases false-positive rates. Therefore, **M3 is confirmed as the primary publication model**, with M4 and M5 serving as exploratory architectural ablations in Section 5.
