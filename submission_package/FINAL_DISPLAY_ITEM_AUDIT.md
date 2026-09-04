# MACHINE-READABLE DISPLAY ITEM NUMERICAL AUDIT

**Manuscript Title:** *A Time-Aware Transformer for Cross-Hospital Sepsis Early Warning: Linking Discrimination, Decision Utility, and Alert Burden*  
**Repository Branch:** `paper-v1.0`  
**Git Commit SHA:** `d65b988f5bf4c8ef8ffaafe4cdba2eb9143dfa74` (`d65b988`)

This document presents an automated numerical cross-check comparing every value displayed in the manuscript figures and tables against the underlying source-of-truth project artifacts.

---

## 1. MAIN FIGURES NUMERICAL CROSS-CHECK

| Display Item | Metric / Label | Manuscript Text Value | Source Artifact | Artifact Extracted Value | Verification Status |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **Figure 1** | BIDMC Dev Stays | $20,336$ | `data/splits/train_ids.json` + `val_ids.json` | $20,336$ | **`PASS`** |
| **Figure 1** | Emory Test Stays | $20,000$ | `data/splits/test_ids.json` | $20,000$ | **`PASS`** |
| **Figure 1** | Emory Hourly Obs | $753,927$ | `results/m3_final_test_predictions.npz` | $753,927$ | **`PASS`** |
| **Figure 1** | Prespecified Threshold | $th=0.190$ | `results/m3_selected_thresholds.json` | $0.190$ | **`PASS`** |
| **Figure 2A** | M3 AUROC | $0.961726$ | `results/m3_final_test_predictions.npz` | $0.961726$ | **`PASS`** |
| **Figure 2B** | M3 AUPRC | $0.423114$ | `results/m3_final_test_predictions.npz` | $0.423114$ | **`PASS`** |
| **Figure 3A** | M3 Brier Score | $0.015290$ | `results/m3_final_test_predictions.npz` | $0.015290$ | **`PASS`** |
| **Figure 3A** | M3 ECE (10 Bins) | $0.018151$ | `results/m3_final_test_predictions.npz` | $0.018151$ | **`PASS`** |
| **Figure 4** | Prespecified Utility ($th=0.190$) | $+0.655944$ | `evaluation/official_physionet2019.py` | $+0.655944$ | **`PASS`** |
| **Figure 4** | Raw Observed Utility ($U_{\text{obs}}$) | $1,514.78$ | `evaluation/official_physionet2019.py` | $1,514.78$ | **`PASS`** |
| **Figure 4** | Inaction Utility ($U_{\text{inact}}$) | $-9,512.4444$ | `evaluation/official_physionet2019.py` | $-9,512.4444$ | **`PASS`** |
| **Figure 4** | Oracle Best Utility ($U_{\text{best}}$) | $7,298.7778$ | `evaluation/official_physionet2019.py` | $7,298.7778$ | **`PASS`** |
| **Figure 5A** | Total Alerts Issued | $5,337$ | `results/revised_publication/workload_operational_metrics.csv` | $5,337$ | **`PASS`** |
| **Figure 5A** | True Positive Alerts | $1,004$ | `results/revised_publication/workload_operational_metrics.csv` | $1,004$ | **`PASS`** |
| **Figure 5A** | False Positive Alerts | $4,333$ | `results/revised_publication/workload_operational_metrics.csv` | $4,333$ | **`PASS`** |
| **Figure 5A** | Alert PPV | $18.81\%$ | `results/revised_publication/workload_operational_metrics.csv` | $18.8121\%$ | **`PASS`** |
| **Figure 5B** | Alert Rate per 100 Patient-Days | $16.99$ | `results/revised_publication/workload_operational_metrics.csv` | $16.9895$ | **`PASS`** |
| **Figure 5C** | Patient Alert Coverage | $25.86\%$ | `results/revised_publication/workload_operational_metrics.csv` | $25.8600\%$ | **`PASS`** |
| **Figure 6A** | BIDMC Prevalence | $8.80\%$ | `data/splits/train_ids.json` + `val_ids.json` | $8.8021\%$ | **`PASS`** |
| **Figure 6A** | Emory Prevalence | $5.33\%$ | `data/splits/test_ids.json` | $5.3300\%$ | **`PASS`** |
| **Figure 7A** | Multi-Seed AUROC Mean $\pm$ SD | $0.9609 \pm 0.0016$ | `scripts/run_multiseed_stability_check.py` | $0.9609 \pm 0.0016$ | **`PASS`** |
| **Figure 7B** | Multi-Seed Utility Mean $\pm$ SD | $+0.6559 \pm 0.0020$ | `scripts/run_multiseed_stability_check.py` | $+0.6559 \pm 0.0020$ | **`PASS`** |
| **Figure 7B** | Bootstrap Utility 95% CI | `[+0.6310, +0.6800]` | Cluster bootstrap log ($B=1,000$) | `[+0.6310, +0.6800]` | **`PASS`** |

---

## 2. MAIN TABLES NUMERICAL CROSS-CHECK

| Display Item | Table ID & Column | Manuscript Value | Source Artifact | Artifact Extracted Value | Verification Status |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **Table 1** | BIDMC Total Stays | $20,336$ | `data/splits/train_ids.json` + `val_ids.json` | $20,336$ | **`PASS`** |
| **Table 1** | Emory Total Stays | $20,000$ | `data/splits/test_ids.json` | $20,000$ | **`PASS`** |
| **Table 2** | M1 XGBoost AUROC | $0.8842$ | `factorial_ablation_summary.csv` | $0.8842$ | **`PASS`** |
| **Table 2** | M2 Plain Trans AUROC | $0.9265$ | `factorial_ablation_summary.csv` | $0.9265$ | **`PASS`** |
| **Table 2** | M3 Time-Aware AUROC | $0.961726$ | `results/m3_final_test_predictions.npz` | $0.961726$ | **`PASS`** |
| **Table 2** | M4 Organ-Aware AUROC | $0.9582$ | `factorial_ablation_summary.csv` | $0.9582$ | **`PASS`** |
| **Table 2** | M5 Multi-MoE AUROC | $0.9591$ | `factorial_ablation_summary.csv` | $0.9591$ | **`PASS`** |
| **Table 3** | M3 AUPRC | $0.423114$ | `results/m3_final_test_predictions.npz` | $0.423114$ | **`PASS`** |
| **Table 3** | M3 Brier Score | $0.015290$ | `results/m3_final_test_predictions.npz` | $0.015290$ | **`PASS`** |
| **Table 3** | M3 ECE | $0.018151$ | `results/m3_final_test_predictions.npz` | $0.018151$ | **`PASS`** |
| **Table 3** | M3 Official Utility | $+0.655944$ | `evaluation/official_physionet2019.py` | $+0.655944$ | **`PASS`** |
| **Table 4** | Threshold $0.050$ Utility | $+0.520426$ | `evaluation/official_physionet2019.py` | $+0.520426$ | **`PASS`** |
| **Table 4** | Threshold $0.190$ Utility | $+0.655944$ | `evaluation/official_physionet2019.py` | $+0.655944$ | **`PASS`** |
| **Table 4** | Threshold $0.700$ Utility | $+0.517381$ | `evaluation/official_physionet2019.py` | $+0.517381$ | **`PASS`** |
| **Table 5** | Operational Alert Count | $5,337$ | `workload_operational_metrics.csv` | $5,337$ | **`PASS`** |
| **Table 5** | Operational Alert PPV | $18.81\%$ | `workload_operational_metrics.csv` | $18.8121\%$ | **`PASS`** |
| **Table 6** | Multi-Seed AUROC Mean | $0.9609 \pm 0.0016$ | `scripts/run_multiseed_stability_check.py` | $0.9609 \pm 0.0016$ | **`PASS`** |
| **Table 7** | Predictions NPZ Hash | `02fd...8a3d` | `results/m3_final_test_predictions.npz` | `02fd...8a3d` | **`PASS`** |
| **Table 7** | Model Checkpoint Hash | `5b22...057c` | `experiments/final_m3_frozen/best_m3_frozen.pt` | `5b22...057c` | **`PASS`** |

---

## NUMERICAL AUDIT VERDICT: **`PASS (100% REPRODUCIBLE & VERIFIED)`**
