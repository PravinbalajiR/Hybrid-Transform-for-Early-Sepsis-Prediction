# COMPLETE PUBLICATION FIGURE AND TABLE MANIFEST

**Manuscript Title:** *A Time-Aware Transformer for Cross-Hospital Sepsis Early Warning: Linking Discrimination, Decision Utility, and Alert Burden*  
**Repository Branch:** `paper-v1.0`  
**Git Commit SHA:** `2df15721c54b670ad337d1d2b7fa1d4f40d866a9` (`2df1572`)

---

## MAIN FIGURES

### Figure 1: Study Design & Two-Stage Threshold Isolation Protocol
- **Description:** Flowchart mapping BIDMC development ($N=20,336$: $18,302$ train / $2,034$ val) $\to$ $M3$ development $\to$ validation threshold selection ($th^*=0.190$) $\to$ frozen checkpoint $\to$ Emory test evaluation ($N=20,000$).
- **Source Artifact:** `data/splits/train_ids.json`, `val_ids.json`, `test_ids.json`, `m3_selected_thresholds.json`
- **Output Files:** [`figures/fig1_study_workflow.png`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/figures/fig1_study_workflow.png), [`figures/fig1_study_workflow.pdf`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/figures/fig1_study_workflow.pdf)
- **Manuscript Location:** Section 3.1 & Section 3.4.

### Figure 2: M3 Architecture & Temporal Early Warning Framing
- **Description:** Schematic of Time-Aware Transformer encoder, input triplet vector $\mathbf{x}(t) = [\mathbf{v}(t), \mathbf{m}(t), \mathbf{\Delta t}(t)]$, Time2Vec delta encoding, $3$ encoder layers, causal real-time risk estimate $p(t)$, and PhysioNet $6$-hour onset shift ($t_{\text{sepsis}} = t_{\text{label}} + 6\text{h}$).
- **Source Artifact:** `tact_model.py`, `best_m3_frozen.pt`
- **Output Files:** [`figures/fig2_model_architecture_temporal.png`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/figures/fig2_model_architecture_temporal.png), [`figures/fig2_model_architecture_temporal.pdf`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/figures/fig2_model_architecture_temporal.pdf)
- **Manuscript Location:** Section 3.3.

### Figure 3: M3 Discrimination Performance (ROC & PR Curves)
- **Description:** Panel A: ROC curve ($M3$ AUROC = $0.961726$ vs. Random $0.5000$). Panel B: Precision-Recall curve ($M3$ AUPRC = $0.423114$ vs. Random prevalence $0.0533$).
- **Source Artifact:** [`results/m3_final_test_predictions.npz`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/results/m3_final_test_predictions.npz)
- **Output Files:** [`figures/fig3_m3_discrimination.png`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/figures/fig3_m3_discrimination.png), [`figures/fig3_m3_discrimination.pdf`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/figures/fig3_m3_discrimination.pdf)
- **Manuscript Location:** Section 4.1.

### Figure 4: M3 Risk Probability Calibration
- **Description:** Calibration diagram showing predicted probability vs. observed event fraction across 10 equal-width bins (Brier Score = `0.015290`, Expected Calibration Error ECE = `0.018151`).
- **Source Artifact:** [`results/m3_final_test_predictions.npz`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/results/m3_final_test_predictions.npz)
- **Output Files:** [`figures/fig4_m3_calibration.png`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/figures/fig4_m3_calibration.png), [`figures/fig4_m3_calibration.pdf`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/figures/fig4_m3_calibration.pdf)
- **Manuscript Location:** Section 4.2.

### Figure 5: Official PhysioNet Utility Across Thresholds
- **Description:** Official PhysioNet 2019 Normalized Utility ($U_{\text{official}}$) across decision thresholds $th \in [0.05, 0.70]$ on Emory test data. Prespecified threshold $th=0.190$ ($U_{\text{official}} = +0.655944$) clearly marked and labeled as descriptive test-set sensitivity sweep.
- **Source Artifact:** [`evaluation/official_physionet2019.py`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/evaluation/official_physionet2019.py), [`results/m3_selected_thresholds.json`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/results/m3_selected_thresholds.json)
- **Output Files:** [`figures/fig5_official_utility_threshold.png`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/figures/fig5_official_utility_threshold.png), [`figures/fig5_official_utility_threshold.pdf`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/figures/fig5_official_utility_threshold.pdf)
- **Manuscript Location:** Section 4.3 & Section 4.4.

### Figure 6: Operational Alert Burden & Workload
- **Description:** Panel A: Total alert composition ($1,004$ TP alerts [$18.81\%$ PPV], $4,333$ FP alerts [$81.19\%$], $5,337$ total alerts). Panel B: Operational alert frequency ($16.99$ alerts per $100$ patient-days). Panel C: Patient coverage ($5,172$ alerted stays [$25.86\%$] vs. $14,828$ non-alerted stays [$74.14\%$]).
- **Source Artifact:** [`results/revised_publication/workload_operational_metrics.csv`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/results/revised_publication/workload_operational_metrics.csv)
- **Output Files:** [`figures/fig6_alert_burden_threshold.png`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/figures/fig6_alert_burden_threshold.png), [`figures/fig6_alert_burden_threshold.pdf`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/figures/fig6_alert_burden_threshold.pdf)
- **Manuscript Location:** Section 4.5.

### Figure 7: Model Family Discriminative Performance
- **Description:** Bar chart comparing AUROC and AUPRC across model progression ($M1$ XGBoost, $M2$ Plain Transformer, $M3$ Time-Aware Transformer, $M4$ Organ-Aware, $M5$ Multi-MoE) on Emory test data. Baseline utility is un-plotted (`N/A`) where raw prediction arrays were not preserved.
- **Source Artifact:** `results/revised_publication/factorial_ablation_summary.csv`
- **Output Files:** [`figures/fig7_model_family_comparison.png`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/figures/fig7_model_family_comparison.png), [`figures/fig7_model_family_comparison.pdf`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/figures/fig7_model_family_comparison.pdf)
- **Manuscript Location:** Section 4.1.

### Figure 8: Cross-Hospital Sepsis Prevalence Shift
- **Description:** Bar chart illustrating sepsis prevalence drop from BIDMC ($8.80\%$) to Emory ($5.33\%$) and explaining why lower external prevalence depresses alert precision (PPV = $18.81\%$).
- **Source Artifact:** `data/splits/train_ids.json`, `test_ids.json`
- **Output Files:** [`figures/fig8_cross_hospital_shift.png`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/figures/fig8_cross_hospital_shift.png), [`figures/fig8_cross_hospital_shift.pdf`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/figures/fig8_cross_hospital_shift.pdf)
- **Manuscript Location:** Section 5.4.

---

## MAIN TABLES

### Table 1: Cohort Characteristics
- **CSV Output:** [`tables/table1_cohort_characteristics.csv`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/tables/table1_cohort_characteristics.csv)
- **LaTeX Output:** [`tables/table1_cohort_characteristics.tex`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/tables/table1_cohort_characteristics.tex)

### Table 2: Cross-Hospital Model Progression ($M1$–$M5$)
- **CSV Output:** [`tables/table2_model_progression.csv`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/tables/table2_model_progression.csv)
- **LaTeX Output:** [`tables/table2_model_progression.tex`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/tables/table2_model_progression.tex)

### Table 3: Final $M3$ External Test Performance Summary
- **CSV Output:** [`tables/table3_final_m3_performance.csv`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/tables/table3_final_m3_performance.csv)
- **LaTeX Output:** [`tables/table3_final_m3_performance.tex`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/tables/table3_final_m3_performance.tex)

### Table 4: Official Threshold Sensitivity Sweep ($th \in [0.05, 0.70]$)
- **CSV Output:** [`tables/table4_threshold_sensitivity.csv`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/tables/table4_threshold_sensitivity.csv)
- **LaTeX Output:** [`tables/table4_threshold_sensitivity.tex`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/tables/table4_threshold_sensitivity.tex)

### Table 5: Multi-Seed Random Initialization Stability ($N=6$ Seeds)
- **CSV Output:** [`tables/table5_multiseed_stability.csv`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/tables/table5_multiseed_stability.csv)
- **LaTeX Output:** [`tables/table5_multiseed_stability.tex`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/tables/table5_multiseed_stability.tex)

### Table 6: Reproducibility Artifact SHA256 Cryptographic Manifest
- **CSV Output:** [`tables/table6_reproducibility_manifest.csv`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/tables/table6_reproducibility_manifest.csv)
- **LaTeX Output:** [`tables/table6_reproducibility_manifest.tex`](file:///C:/Users/gokul/Desktop/sepsis%20prediction/Sepsis-Hybrid-Transformer/tables/table6_reproducibility_manifest.tex)
