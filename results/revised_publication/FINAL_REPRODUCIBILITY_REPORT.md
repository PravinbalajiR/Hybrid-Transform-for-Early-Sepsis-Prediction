# 🔬 FINAL REPRODUCIBILITY REPORT: M1–M5 SEPSIS STUDY

**Repository:** `Hybrid-Transform-for-Early-Sepsis-Prediction`  
**Branch:** `paper-v1.0`  
**Git Commit SHA:** `dc8caea`  
**Evaluation Engine:** Official PhysioNet 2019 Evaluator (`evaluation/official_physionet2019.py`)  
**Environment:** Python 3.11, PyTorch 2.1+, NumPy 1.24+, Pandas 2.0+, CUDA enabled.

---

## 1. Primary Model Artifacts & Provenance Hashes

- **Frozen Checkpoint:** `experiments/final_m3_frozen/best_m3_frozen.pt`  
  - SHA256: `5b22607444f4a242a52d0d9337e60c4c63044542dc6796a4a9de78c5ef38057c`
- **Emory Test Predictions NPZ:** `results/m3_final_test_predictions.npz`  
  - SHA256: `e4a6a5e171b3e94bd2d6b38c2ef40eb14032d91c1b3f9ffc129e9ae70678ed70`
- **Test Split Manifest:** `data/splits/test_ids.json`  
  - SHA256: `55d5bc58000bc19e59d9eef27ca5f5d81bdab7ed74a88f7b764c0173adbd923b`

---

## 2. Authoritative External Test Results (Emory Test Set B, N=20,000)

- **Development Cohort (Set A):** Beth Israel Deaconess Medical Center (BIDMC / Hospital A) ($N = 20,336$ ICU stays: $16,192$ train, $4,144$ val).
- **External Test Cohort (Set B):** Emory University Hospital (Hospital B) ($N = 20,000$ ICU stays: $1,066$ septic, $18,934$ non-septic).
- **Transfer Direction:** **BIDMC $\to$ Emory**.
- **Prespecified Decision Threshold:** **`0.190`** (Selected on BIDMC validation set before external test evaluation).
- **Test Performance Metrics:**
  - **AUROC:** **`0.961726`** (`0.9617`)
  - **AUPRC:** **`0.423114`** (`0.4231`)
  - **Accuracy:** **`0.971542`** (`97.15%`)
  - **F-measure:** **`0.231804`**
  - **Official Normalized Utility ($U_{\text{official}}$):** **`0.655944`** (`+0.6559`)
  - **Ground-Truth Oracle Ceiling ($U_{\text{oracle}}$):** **`1.000000`** ($100\%$ Max Utility)
  - **Brier Score:** `0.015290`
  - **Expected Calibration Error (ECE):** `0.018151`

---

## 3. Operational Workload Metrics

- **Total Alerts Issued:** $5,337$ alerts ($1,004$ True Sepsis Alerts, $4,333$ Non-Sepsis False Alerts)
- **Total ICU Patient-Days:** $31,413.6$ patient-days ($753,927$ hourly observations / $24.0$ hours)
- **Alert Frequency:** **$16.99$ alerts per 100 patient-days**
- **Alert Positive Predictive Value (PPV):** **$18.81\%$**
- **Alerts per Patient:** **$0.267$ alerts / patient**
- **False Alerts per Non-Septic Patient:** **$0.229$ false alerts / patient**
- **Percentage of Patients Alerted:** **$25.86\%$**

---

## 4. Multi-Seed Stability Summary ($N=6$ Seeds)

- **AUROC:** $0.9609 \pm 0.0016$
- **AUPRC:** $0.4224 \pm 0.0026$
- **Official Utility ($th=0.190$):** $+0.6559 \pm 0.0020$

---

## 5. Limitations & Unsupported Claims Removed

1. **Test-Set Threshold Optimization Removed:** The prespecified threshold $0.190$ is treated as the sole deployable result; threshold sweeps are reported strictly as post-hoc sensitivity analyses.
2. **Unsupported Baseline Utilities:** Baseline models without saved prediction NPZ files have their utility values displayed as `—` in official benchmark tables.
3. **No Overclaiming:** Terms like "state-of-the-art", "proves", and "clinically ready" are strictly excluded.
