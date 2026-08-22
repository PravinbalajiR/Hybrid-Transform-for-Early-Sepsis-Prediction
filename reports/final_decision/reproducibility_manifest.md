# 🔐 REPRODUCIBILITY MANIFEST (TASK 11 & 13)

This manifest records all exact SHA256 cryptographic hashes for datasets, frozen checkpoints, test prediction artifacts, and codebase scripts to ensure zero-drift scientific reproducibility.

---

## 1. Core Model & Artifact Cryptographic Hashes

| Artifact / File Name | File Path | SHA256 Cryptographic Hash |
| :--- | :--- | :--- |
| **Frozen M3 Checkpoint** | `experiments/final_m3_frozen/best_m3_frozen.pt` | `5b22607444f4a242a52d0d9337e60c4c63044542dc6796a4a9de78c5ef38057c` |
| **BIDMC Test Predictions NPZ** | `results/m3_final_test_predictions.npz` | `e4a6a5e171b3e94bd2d6b38c2ef40eb14032d91c1b3f9ffc129e9ae70678ed70` |
| **Emory Val Predictions NPZ** | `results/m3_final_val_predictions.npz` | `c3ee258be16e11894d38e219fb099fef4c0dceaeedec0b37493a778b4a7ee5f7` |
| **Patient Split Manifest** | `data/splits/test_ids.json` | `55d5bc58000bc19e59d9eef27ca5f5d81bdab7ed74a88f7b764c0173adbd923b` |
| **Independent Utility Calculator** | `scripts/oracle_reconciliation_independent.py` | `0f0bf6085a815a5fbc4001923cb82dc99a2fbde6097561f7481a53a9cd388a10` |

---

## 2. Model Performance Metrics on BIDMC Test Cohort

- **Held-out Test Cohort Size:** $N = 20,000$ patients ($1,066$ septic, $18,934$ non-septic)
- **Total Hourly Observations:** $753,927$ hours ($27,000$ septic hours, $726,927$ non-septic hours)
- **AUROC:** `0.961726` (reported as `0.9617`)
- **AUPRC:** `0.423114` (reported as `0.4231`)
- **Brier Score:** `0.015290` (raw probability score) / `0.021326` (uncalibrated logit scale)
- **Expected Calibration Error (ECE):** `0.018151`
- **Deployable Utility (`FROZEN_MODEL_UTILITY`):** `-0.257312450379` ($th = 0.190, C = 36\text{h}$)

---

## 3. Machine Environment & Seeding
- **Primary Random Seed:** `42`
- **PyTorch Seeding:** `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`
- **NumPy Seeding:** `np.random.seed(42)`
- **Bootstrap Resamples:** $B = 1,000$ (patient-level resampling with replacement)
