# Experiment Tracking Log

| ID | Model | Learning Rate | Batch Size | Epochs | Utility Score | AUROC | AUPRC | Notes |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| EXP-000 | XGBoost Baseline | N/A | Static | 300 trees | -2.0000 | 0.7708 | 0.0497 | Static aggregate baseline (mean/std/min/max/last/count), vectorized feature extraction |
| EXP-001 | Plain Transformer | 1e-4 | 32 | 30 | -0.9233 | 0.9601 | 0.4129 | Naive mean-imputed input, positional encoding |
| EXP-002 | Time-Aware Transformer | 1e-4 | 32 | 30 | -0.9932 | 0.9697 | 0.4851 | Value + Mask + Time-Delta triplet input |
| EXP-003 | Grouped MLP Organ Branch | 1e-4 | 32 | 30 | TBD | TBD | TBD | Simplified organ-grouped MLP before Transformer |
| EXP-004 | Full Hybrid Model | 1e-4 | 32 | 30 | TBD | TBD | TBD | Dual-branch + Cross-Attention Fusion |
