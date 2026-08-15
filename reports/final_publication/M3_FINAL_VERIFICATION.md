# M3 Primary Model Read-Only Verification

**Model:** Time-Aware Transformer (TACTModel)  
**Checkpoint Path:** `experiments/final_m3_frozen/best_m3_frozen.pt`  
**Checkpoint SHA256:** `5b22607444f4a242a52d0d9337e60c4c63044542dc6796a4a9de78c5ef38057c`  

---

## 1. Authoritative Performance Metrics

- **AUROC**: `0.9617` (95% CI: `[0.9495, 0.9727]`)
- **AUPRC**: `0.4231` (95% CI: `[0.3359, 0.5185]`)
- **F1 Score**: `0.4110`
- **Precision**: `0.3099`
- **Recall**: `0.6103`
- **ECE**: `0.0407`
- **Mean Lead Time**: `5.7 hours` (95% CI: `[5.0h, 6.5h]`)
- **$\ge$6h Early Warning**: `37.6%`
- **$\ge$1h Early Warning**: `56.5%`
- **False Positive Rate / Hour**: `1.83%`
- **PhysioNet Utility**: `-0.9535`
- **Operating Threshold**: `0.60` (Validation-Locked)
