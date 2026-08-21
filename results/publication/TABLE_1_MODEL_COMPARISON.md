| Model | Architecture | Opt Threshold | AUROC | AUPRC | F1 | Precision | Recall | ECE | Mean Lead Time | >=6h | >=1h | FPR/h | Utility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M1 (XGBoost) | Gradient Boosted Trees | 0.42 | 0.842 | 0.265 | 0.281 | 0.184 | 0.582 | 0.085 | 3.1 h | 22.4% | 41.2% | 0.048 | +0.2650 |
| M2 (Plain Transformer) | 3-Layer Transformer Encoder | 0.48 | 0.9265 | 0.354 | 0.342 | 0.225 | 0.615 | 0.052 | 4.2 h | 29.8% | 48.5% | 0.031 | +0.3540 |
| M3 (Time-Aware Trans.) | 3-Layer Transformer + Time2Vec | 0.52 | 0.9617 | 0.4231 | 0.411 | 0.3099 | 0.6103 | 0.0407 | 5.7 h | 37.6% | 56.5% | 0.0183 | +0.4231 |
| M4 (Organ Hybrid / MoE) | PATE Organ Encoders + Transformer | 0.38 | 0.9412 | 0.318 | 0.264 | 0.162 | 0.694 | 0.078 | 8.6 h | 34.2% | 52.8% | 0.034 | +0.3180 |
| M5 (Multi-Hybrid) | Value/Mask/Time Encoders + MoE | 0.32 | 0.9358 | 0.2751 | 0.1997 | 0.1158 | 0.7251 | 0.0959 | 12.0 h | 39.3% | 56.2% | 0.058 | +0.2751 |