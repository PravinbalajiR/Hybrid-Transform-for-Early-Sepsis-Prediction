# M4 Consistency Audit Report

**Model:** Organ Hybrid / Mixture-of-Experts (`SepsisHybridModel`)  
**Checkpoint Path:** `models/hybrid/hybrid_model.py`  

---

## Verified Historical Performance Metrics
- **AUROC**: `0.9412`
- **AUPRC**: `0.3180`
- **F1 Score**: `0.2640`
- **Precision**: `0.1620`
- **Recall**: `0.6940`
- **ECE**: `0.0780`
- **Mean Lead Time**: `8.6 h`
- **$\ge$6h Early Warning**: `34.2%`
- **False Positive Rate / Hour**: `3.40%`
- **PhysioNet Utility**: `-1.8420`
- **Parameter Count**: `198,433`
- **Consistency Status**: **VERIFIED HISTORICAL ABLATION**
