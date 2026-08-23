# REPRODUCIBILITY MANIFEST & CRYPTOGRAPHIC LEDGER

**Repository:** `Hybrid-Transform-for-Early-Sepsis-Prediction`  
**Branch:** `paper-v1.0`  
**Git Commit SHA:** `16f189f76a5960d7042a98e826b1bc3128b9c2fb` (`16f189f`)  
**Manuscript Title:** *A Time-Aware Transformer for Cross-Hospital Sepsis Early Warning: Linking Discrimination, Decision Utility, and Alert Burden*

---

## 1. FROZEN ARTIFACT SHA256 DIGEST

| Artifact Relative Path | File Size | SHA256 Cryptographic Hash | Verification Status |
| :--- | :---: | :--- | :---: |
| `results/m3_final_test_predictions.npz` | $2,799.5$ KB | `02fd6eb78682be8ca5743c4b3fddfcc7f57ed56f27f8496092108c30b2188a3d` | **MATCHES** |
| `experiments/final_m3_frozen/best_m3_frozen.pt` | $1,551.2$ KB | `5b22607444f4a242a52d0d9337e60c4c63044542dc6796a4a9de78c5ef38057c` | **MATCHES** |
| `data/splits/train_ids.json` | $250.2$ KB | `06edf3b5519abdaee736da14763ffaf45226523ad62a658947f74a55ee4121d2` | **MATCHES** |
| `data/splits/val_ids.json` | $27.8$ KB | `71bb23f3b5aef82c9169dc97a7687e8ae454756ef97866e51f1a75d32dbeb15a` | **MATCHES** |
| `data/splits/test_ids.json` | $273.4$ KB | `f7932a915251dd22554493ee7b9a18a0241d6805e0fca2b85021c5750648d00f` | **MATCHES** |
| `evaluation/official_physionet2019.py` | $17.8$ KB | `d0f65da3d42ce68cad80e290050bce4b8f2efc7ad3f13c0a1f70a331fbd8ff06` | **MATCHES** |
| `scripts/run_multiseed_stability_check.py` | $14.9$ KB | `bab4ba342f010b75c824edc312e82aae508711ab2fcb9725b3f33dfe10389be6` | **MATCHES** |
| `results/revised_publication/factorial_ablation_summary.csv` | $0.3$ KB | `05d0cd92bedfb950d56ceccec1c16ef74d2ab243eb83475fd900e37d9b5a3de8` | **MATCHES** |
| `results/revised_publication/workload_operational_metrics.csv` | $0.3$ KB | `4e4039b1d003bb1fe8698b6d8cbfb481cee6177a148da12a94423b05c7178153` | **MATCHES** |

---

## 2. KEY EXPERIMENTAL PARAMETERS & METRICS

- **BIDMC Development Cohort Stays:** $20,336$ ($18,302$ train [$90.0\%$], $2,034$ val [$10.0\%$])
- **Emory External Test Cohort Stays:** $20,000$ ($1,066$ septic [$5.33\%$], $18,934$ non-septic [$94.67\%$], $753,927$ hourly observations)
- **$M3$ Discriminative Performance:** AUROC = $0.961726$ ($0.9617$), AUPRC = $0.423114$ ($0.4231$)
- **Calibration Metrics:** Brier Score = $0.015290$, ECE = $0.018151$ (10 equal-width bins)
- **Prespecified Validation Decision Threshold:** $th = 0.190$ (locked strictly on BIDMC validation split $N=2,034$)
- **Official Utility Components:** $U_{\text{obs}} = 1,515.6500$ pts, $U_{\text{inact}} = -9,512.4444$ pts, $U_{\text{best}} = 7,298.7778$ pts
- **Official Normalized Utility ($U_{\text{official}}$):** $\mathbf{+0.655944}$ ($+0.6559$, 95% CI: `[+0.6310, +0.6800]`)
- **Operational Workload Metrics ($th=0.190$):** $5,337$ total alerts ($1,004$ TP, $4,333$ FP), PPV = $18.81\%$, Alert Frequency = $16.99$ alerts/100 patient-days, Patient Coverage = $25.86\%$ ($5,172$/$20,000$ stays)
- **Multi-Seed Stability ($N=6$ seeds):** AUROC = $0.9609 \pm 0.0016$, Utility = $+0.6559 \pm 0.0020$
