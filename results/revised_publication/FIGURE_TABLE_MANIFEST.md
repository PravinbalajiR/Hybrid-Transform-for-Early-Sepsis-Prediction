# PUBLICATION FIGURE AND TABLE MANIFEST

**Manuscript Title:** *A Time-Aware Transformer for Cross-Hospital Sepsis Early Warning: Linking Discrimination, Decision Utility, and Alert Burden*  
**Repository Branch:** `paper-v1.0`  
**Git Commit SHA:** `41d9417fa3bfe7735398243be1297ae4a434bbbb` (`41d9417`)

---

## MAIN FIGURES

### Figure 1: Discrimination Performance (ROC & PR Curves)
- **Title:** Discrimination performance of the final $M3$ model on the external Emory test cohort ($N=20,000$).
- **Source Artifact:** [`results/m3_final_test_predictions.npz`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/results/m3_final_test_predictions.npz)
- **Generation Script:** [`scripts/generate_publication_figures.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/generate_publication_figures.py)
- **Output Files:**
  - Raster (PNG 300 DPI): [`figures/figure1_discrimination.png`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/figures/figure1_discrimination.png)
  - Vector (PDF): [`figures/figure1_discrimination.pdf`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/figures/figure1_discrimination.pdf)
  - Vector (SVG): [`figures/figure1_discrimination.svg`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/figures/figure1_discrimination.svg)
- **Verified Values:** AUROC = `0.961726` (`0.9617`), AUPRC = `0.423114` (`0.4231`).
- **Manuscript Location:** Section 4.1 & Section 4.2.

### Figure 2: Official Decision Utility Across Thresholds
- **Title:** Official PhysioNet 2019 normalized utility across decision thresholds on the external Emory test cohort ($N=20,000$).
- **Source Artifact:** [`evaluation/official_physionet2019.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/evaluation/official_physionet2019.py), [`results/m3_selected_thresholds.json`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/results/m3_selected_thresholds.json)
- **Generation Script:** [`scripts/generate_publication_figures.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/generate_publication_figures.py)
- **Output Files:**
  - Raster (PNG 300 DPI): [`figures/figure2_utility_threshold.png`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/figures/figure2_utility_threshold.png)
  - Vector (PDF): [`figures/figure2_utility_threshold.pdf`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/figures/figure2_utility_threshold.pdf)
  - Vector (SVG): [`figures/figure2_utility_threshold.svg`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/figures/figure2_utility_threshold.svg)
- **Verified Values:** Prespecified operating threshold $th=0.190$ ($U_{\text{official}} = +0.655944$). Threshold sweep ($th \in [0.05, 0.70]$) explicitly labeled as descriptive sensitivity analysis.
- **Manuscript Location:** Section 4.3 & Section 4.4.

### Figure 3: Operational Alert Burden & Workload
- **Title:** Operational alert burden of the final $M3$ model on the external Emory test cohort ($N=20,000$).
- **Source Artifact:** [`results/revised_publication/workload_operational_metrics.csv`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/results/revised_publication/workload_operational_metrics.csv)
- **Generation Script:** [`scripts/generate_publication_figures.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/scripts/generate_publication_figures.py)
- **Output Files:**
  - Raster (PNG 300 DPI): [`figures/figure3_alert_burden.png`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/figures/figure3_alert_burden.png)
  - Vector (PDF): [`figures/figure3_alert_burden.pdf`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/figures/figure3_alert_burden.pdf)
  - Vector (SVG): [`figures/figure3_alert_burden.svg`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/figures/figure3_alert_burden.svg)
- **Verified Values:** Total alerts = $5,337$ ($1,004$ TP, $4,333$ FP), PPV = $18.81\%$, Alert frequency = $16.99$ alerts/100 patient-days, Patient coverage = $25.86\%$.
- **Manuscript Location:** Section 4.5.

---

## MAIN TABLES

### Table 1: Cross-Hospital Model Family Progression
- **Title:** Cross-Hospital Performance Comparison Across Model Family (Emory External Test Set, $N=20,000$).
- **Source Artifact:** `results/revised_publication/factorial_ablation_summary.csv`
- **Verified Content:**
  - $M1$ (XGBoost): AUROC = $0.8842$, AUPRC = $0.2851$, Brier = $0.0241$, ECE = $0.0382$, Utility = `—`
  - $M2$ (Plain Transformer): AUROC = $0.9265$, AUPRC = $0.3412$, Brier = $0.0189$, ECE = $0.0245$, Utility = `—`
  - **$M3$ (Time-Aware Transformer):** AUROC = **$0.9617$**, AUPRC = **$0.4231$**, Brier = **$0.0153$**, ECE = **$0.0182$**, Utility = **$+0.6559$**
  - $M4$ (Organ-Aware Hybrid): AUROC = $0.9582$, AUPRC = $0.4150$, Brier = $0.0158$, ECE = $0.0195$, Utility = `—`
  - $M5$ (Multi-Hybrid / MoE): AUROC = $0.9591$, AUPRC = $0.4182$, Brier = $0.0156$, ECE = $0.0190$, Utility = `—`
- **Manuscript Location:** Section 4.1.

### Table 2: Official Threshold Sensitivity Analysis
- **Title:** Official Threshold Sensitivity Sweep (Emory External Test Set, $N=20,000$).
- **Source Artifact:** `results/m3_selected_thresholds.json`, `evaluate_sepsis_score.py`
- **Verified Content:** Thresholds $th \in [0.05, 0.70]$, with $th=0.190$ marked as prespecified operating threshold ($U_{\text{official}} = +0.655944$).
- **Manuscript Location:** Section 4.4.
