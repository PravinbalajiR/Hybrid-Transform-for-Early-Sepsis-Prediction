# COMPREHENSIVE FIGURE AND TABLE SCIENTIFIC AUDIT

**Manuscript Title:** *A Time-Aware Transformer for Cross-Hospital Sepsis Early Warning: Linking Discrimination, Decision Utility, and Alert Burden*  
**Repository Branch:** `paper-v1.0`  
**Git Commit SHA:** `b1a361bf7000d6efc192bf88e1aa0eb5a539db14` (`b1a361b`)

---

## 1. GENERATED FIGURE MANIFEST & PLACEMENT RECOMMENDATION

| Figure ID | Figure Title | Source Artifact | Metrics Plotted | Placement Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| **Figure 1** | Study Design & Temporal Early-Warning Framework | Split JSONs, `m3_selected_thresholds.json` | Cohort sizes ($20,336$ dev, $20,000$ test), threshold isolation ($th=0.190$) | **Main Text (Section 1/3)** |
| **Figure 2** | M1–M5 Model Family Architectural Progression | `tact_model.py`, model code | Layer specs ($d_{\text{model}}=64$, params $185\text{K}$–$450\text{K}$) | **Main Text (Section 3)** |
| **Figure 3** | M3 Discrimination Performance (ROC & PR Curves) | `m3_final_test_predictions.npz` | AUROC ($0.961726$), AUPRC ($0.423114$) | **Main Text (Section 4.1)** |
| **Figure 4** | M3 Calibration & Risk Distribution | `m3_final_test_predictions.npz` | Brier ($0.015290$), ECE ($0.018151$), Risk histogram | **Main Text (Section 4.2)** |
| **Figure 5** | Four-Panel Official Utility & Threshold Sensitivity | `official_physionet2019.py`, threshold sweep | $U_{\text{official}}$ ($+0.655944$), PPV, Alert Rate, Coverage vs $th$ | **Main Text (Section 4.3/4.4)** |
| **Figure 6** | Operational Alert Burden at Prespecified Threshold 0.190 | `workload_operational_metrics.csv` | $5,337$ alerts, $18.81\%$ PPV, $16.99$ alerts/100 days, $25.86\%$ coverage | **Main Text (Section 4.5)** |
| **Figure 7** | Cross-Hospital Sepsis Prevalence & Alert PPV Shift | `train_ids.json`, `test_ids.json` | Prevalence ($8.80\%$ to $5.33\%$), AUROC, Alert PPV | **Supplementary / Main (Section 5.4)** |
| **Figure 8** | Model-by-Model Performance Comparison | `factorial_ablation_summary.csv` | AUROC, AUPRC, Brier across $M1$–$M5$ | **Supplementary (Section 4.1)** |
| **Figure 9** | Multi-Seed Stability & Bootstrap Uncertainty | Multi-seed logs, bootstrap arrays | AUROC ($0.9609 \pm 0.0016$), Utility ($+0.6559 \pm 0.0020$, CI `[+0.6310, +0.6800]`) | **Supplementary (Section 4.9)** |
| **Figure 10**| Factorial Component Ablation Main Effects | `factorial_ablation_summary.csv` | Mask effect ($+0.0155$), Delta effect ($+0.0215$) | **Supplementary (Section 4.6)** |

---

## 2. GENERATED TABLE MANIFEST & PLACEMENT RECOMMENDATION

| Table ID | Table Title | Source Artifact | Format Available | Placement Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| **Table 1** | Cohort Characteristics & Dataset Split Summary | `train_ids.json`, `val_ids.json`, `test_ids.json` | CSV & TeX | **Main Text (Section 3.1)** |
| **Table 2** | Cross-Hospital Model Progression ($M1$–$M5$) | Model code specifications | CSV & TeX | **Main Text (Section 3.3/4.1)** |
| **Table 3** | Main Predictive Performance & Official Utility | `m3_final_test_predictions.npz`, `official_physionet2019.py` | CSV & TeX | **Main Text (Section 4.1/4.3)** |
| **Table 4** | Operational Performance Breakdown at $th=0.190$ | `workload_operational_metrics.csv` | CSV & TeX | **Main Text (Section 4.5)** |
| **Table 5** | Official Threshold Sensitivity Sweep ($th \in [0.05, 0.70]$) | `m3_selected_thresholds.json` | CSV & TeX | **Supplementary (Section 4.4)** |
| **Table 6** | Factorial Component Ablation Results | `factorial_ablation_summary.csv` | CSV & TeX | **Supplementary (Section 4.6)** |
| **Table 7** | Multi-Seed Random Initialization Stability ($N=6$ Seeds) | Multi-seed script logs | CSV & TeX | **Supplementary (Section 4.9)** |

---

## 3. SCIENTIFIC VALIDATION CHECKS

1. **Figure/Table Existence:** All 10 figures exist as high-resolution PNG (300 DPI) and vector PDF in `figures/`, `results/revised_publication/figures/`, and `submission_package/figures/`. All 7 tables exist as CSV and TeX in `tables/` and `results/revised_publication/tables/`.
2. **Numerical Consistency:** Every plotted and tabulated value matches the verified source artifacts 100%.
3. **Threshold Isolation:** $th=0.190$ is explicitly highlighted and labeled as prespecified on BIDMC validation. Test-set sweeps are marked strictly as descriptive sensitivity analysis.
4. **No Synthetic Data:** Zero synthetic or fabricated points were introduced. Baseline utilities without preserved prediction arrays are represented strictly as `—`.
5. **Frozen Artifact Integrity:** All 9 source-of-truth cryptographic SHA256 hashes remain 100% unchanged.
