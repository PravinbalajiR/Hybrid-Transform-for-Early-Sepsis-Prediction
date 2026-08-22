# 🗺️ MANUSCRIPT EVIDENCE MAP

This document maps every core manuscript claim to its empirical evidence, numerical result, figure/table location, and authoritative source artifact.

---

## 1. Core Manuscript Evidence Map

| Manuscript Claim | Empirical Evidence | Verified Numerical Result | Figure / Table Location | Source Artifact File |
| :--- | :--- | :---: | :---: | :--- |
| **Claim 1:** $M3$ provides the strongest discriminative representation among the model family. | Comparative benchmarking of $M1$–$M5$, GRU-D, TCN on Emory test set ($N=20,000$). | AUROC = `0.9617`, AUPRC = `0.4231` (vs. XGBoost `0.8842`, GRU-D `0.9415`). | Table 2 & Figure 2 | `results/revised_publication/extended_baselines_summary.csv` |
| **Claim 2:** Explicit missingness masks ($m$) and time deltas ($\Delta t$) drive discriminative gains. | $2 \times 2$ Factorial component ablation across 5 random seeds. | Main effect $m = +0.0155$, Main effect $\Delta t = +0.0215$ AUROC. | Table 3 & Figure 3 | `results/revised_publication/factorial_ablation_summary.csv` |
| **Claim 3:** Increasing architectural complexity beyond $M3$ ($M4, M5$) yields no further gains. | Organ-aware ($M4$) and multi-hybrid routing ($M5$) models compared against $M3$. | $M4$ AUROC = `0.9582`, $M5$ AUROC = `0.9591` (vs. $M3$ `0.9617`). | Table 2 & Figure 2 | `results/revised_publication/extended_baselines_summary.csv` |
| **Claim 4:** High AUROC does not guarantee positive net clinical utility under cross-hospital deployment. | Deployable evaluation at prespecified validation policy ($th=0.190, C=36\text{h}$). | `FROZEN_MODEL_UTILITY` = **`-0.2573`** (95% CI: `[-0.2828, -0.2335]`). | Table 5 & Figure 4 | `results/m3_final_test_predictions.npz` |
| **Claim 5:** Positive utility is mathematically attainable under an infeasible label-informed upper bound. | Zero-dependency computation using true labels $y_{\text{true}}$ and optimal timing only. | `GROUND_TRUTH_ORACLE_CEILING` = **`+0.8262`** (95% CI: `[+0.8067, +0.8448]`). | Table 5 & Figure 5 | `reports/oracle_reconciliation/source_inventory.md` |
| **Claim 6:** Global score policies remain strictly negative even under optimal hindsight tuning. | 2D policy sweep across thresholds $th \in [0.005, 0.995]$ and Cooldowns $C \in [6, 336]\text{h}$. | `HINDSIGHT_GRID_SCORE_POLICY_CEILING` = **`-0.1983`** (Peak at $C=72\text{h}$). | Table 5 & Figure 4 | `results/oracle_reconciliation/extended_cooldown_grid.csv` |
| **Claim 7:** Negative deployable utility is robust across random initialization seeds. | Training $M3$ across 6 distinct seeds from scratch. | Utility = $-0.2573 \pm 0.0020$, AUROC = $0.9609 \pm 0.0016$ ($0$ positive runs). | Table 4 | `results/multiseed/multiseed_summary.csv` |
| **Claim 8:** Adaptive threshold requirements cannot be reliably predicted from early observable features. | Locked predictability pipeline trained on BIDMC, evaluated ONCE on Emory. | Test AUPRC = `0.2653` (vs. naive base rate `0.2608`), AUROC = `0.5057`. | Figure 6 | `reports/oracle_reconciliation/adaptive_threshold_predictability.md` |
| **Claim 9:** The `ORACLE_TO_GLOBAL_POLICY_UTILITY_GAP` is statistically significant. | Paired patient-level bootstrap resampling ($B=1,000$ iterations). | $\Delta = \mathbf{+1.0246}$ (95% CI: `[+0.9997, +1.0494]`, $p < 0.0001$). | Table 5 & Figure 5 | `results/oracle_reconciliation/paired_significance_tests.csv` |
| **Claim 10:** Prespecified policy issues $16.99$ alerts per $100$ patient-days with $18.81\%$ PPV. | Operational workload audit across $753,927$ hourly observations on Emory test data. | $5,337$ alerts issued ($1,004$ TP, $4,333$ FP), PPV = `18.81%`. | Table 4 & Figure 4 | `results/revised_publication/workload_operational_metrics.csv` |
