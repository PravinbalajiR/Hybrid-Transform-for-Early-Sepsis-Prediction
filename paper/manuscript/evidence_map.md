# 🗺️ MANUSCRIPT EVIDENCE MAP (OFFICIAL PHYSIONET 2019 METRIC)

This document maps every core manuscript claim to its empirical evidence, numerical result, figure/table location, and authoritative source artifact under the official PhysioNet 2019 metric (`evaluate_sepsis_score.py`).

---

## 1. Core Manuscript Evidence Map

| Manuscript Claim | Empirical Evidence | Verified Numerical Result | Figure / Table Location | Source Artifact File |
| :--- | :--- | :---: | :---: | :--- |
| **Claim 1:** $M3$ provides the strongest discriminative representation among the model family. | Comparative benchmarking of $M1$–$M5$, GRU-D, TCN on Emory test set ($N=20,000$). | AUROC = `0.9617`, AUPRC = `0.4231` (vs. XGBoost `0.8842`, GRU-D `0.9415`). | Table 2 & Figure 2 | `results/m3_final_test_predictions.npz` |
| **Claim 2:** Explicit missingness masks ($m$) and time deltas ($\Delta t$) drive discriminative gains. | $2 \times 2$ Factorial component ablation across 5 random seeds. | Main effect $m = +0.0155$, Main effect $\Delta t = +0.0215$ AUROC. | Table 3 & Figure 3 | `results/revised_publication/factorial_ablation_summary.csv` |
| **Claim 3:** Increasing architectural complexity beyond $M3$ ($M4, M5$) yields no further gains. | Organ-aware ($M4$) and multi-hybrid routing ($M5$) models compared against $M3$. | $M4$ AUROC = `0.9582`, $M5$ AUROC = `0.9591` (vs. $M3$ `0.9617`). | Table 2 & Figure 2 | `honest_official_physionet2019_baselines.csv` |
| **Claim 4:** $M3$ achieves a strong Official Normalized Utility score under cross-hospital deployment. | Official evaluation using `evaluate_sepsis_score` at validation threshold $th=0.190$. | `FROZEN_MODEL_UTILITY` = **`+0.6559`** (`+0.655944`, 95% CI: `[+0.6310, +0.6800]`). | Table 5 & Figure 4 | `evaluate_sepsis_score` on Emory test PSVs |
| **Claim 5:** Prespecified threshold $th=0.190$ is the utility-maximizing operating point on the external test set. | Official PhysioNet utility sweep across thresholds $th \in [0.05, 0.70]$ on Emory test data. | Peak utility = **`+0.6559`** at $th=0.190$. | Table 5 & Figure 4 | `official_threshold_sensitivity_sweep.csv` |
| **Claim 6:** Official normalized ground-truth oracle ceiling is $1.000000$. | Official evaluation using true labels and optimal timing window $[t_{\text{sepsis}}-12, t_{\text{sepsis}}+3]$. | `GROUND_TRUTH_ORACLE_CEILING` = **`1.000000`** ($100\%$ Max Utility). | Table 5 & Figure 5 | `evaluate_sepsis_score` on Emory test PSVs |
| **Claim 7:** High utility is seed-stable across random initialization seeds. | Training $M3$ across 6 distinct seeds from scratch. | Utility = $+0.6559 \pm 0.0020$, AUROC = $0.9609 \pm 0.0016$. | Table 4 | `results/multiseed/multiseed_summary.csv` |
| **Claim 8:** Prespecified policy issues $16.99$ alerts per $100$ patient-days with $18.81\%$ PPV. | Operational workload audit across $753,927$ hourly observations on Emory test data. | $5,337$ alerts issued ($1,004$ TP, $4,333$ FP), PPV = `18.81%`. | Table 4 & Figure 4 | `results/revised_publication/workload_operational_metrics.csv` |
