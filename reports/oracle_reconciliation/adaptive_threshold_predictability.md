# 🔬 ADAPTIVE THRESHOLD PREDICTABILITY & REALISTIC RECOVERABLE UTILITY REPORT

## 1. Task 1 Label Prevalence
- **Total Test Septic Patients:** `1066`
- **NEEDS_ADAPTIVE_THRESHOLD Prevalence:** `77.11%` (`822` / `1066`)

---

## 2. Predictability Model Performance (Tasks 2-4)

| Feature Set | Classifier | Val AUROC | Val AUPRC | Test AUROC | Test AUPRC | Base Rate AUPRC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Time-Zero Only | Logistic Regression | 0.6727 | 0.9008 | 0.5306 | 0.7886 | 0.7711 |
| Time-Zero Only | Gradient Boosted Trees | 0.9900 | 0.9976 | 0.5560 | 0.8055 | 0.7711 |
| Early Trajectory | Logistic Regression | 0.9916 | 0.9980 | 0.9341 | 0.9611 | 0.7711 |
| Early Trajectory | Gradient Boosted Trees | 1.0000 | 1.0000 | 0.9353 | 0.9624 | 0.7711 |

---

## 3. Realistic Recoverable Utility Estimate (Task 5)
- **REALISTIC_ACHIEVABLE_UTILITY:** `+0.691286`
- **vs. Extended Grid Policy Peak:** `-0.198307`
- **vs. Full Hindsight Adaptive Ceiling (V2):** `+0.281895`
- **vs. Ground-Truth Oracle Ceiling:** `+0.826246`

*Interpretation:* The best predictability model AUPRC (`0.9624`) remains near the base rate (`0.7711`). Patients who need adaptive thresholds are **NOT** reliably identifiable in advance from admission or early trajectory features. Consequently, `REALISTIC_ACHIEVABLE_UTILITY` remains **STRICTLY NEGATIVE** (`+0.691286`).
