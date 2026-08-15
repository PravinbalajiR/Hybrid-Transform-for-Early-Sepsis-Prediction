# M3 Component Ablation Checkpoint Selection & Verification

**Audit Date:** 2026-08-15  
**Primary Reference Model:** M3-Full (`experiments/final_m3_frozen/best_m3_frozen.pt`)  

---

## 1. Selected Checkpoints by Variant

1. **M3-Full (Primary Model)**:
   - **Path**: `experiments/final_m3_frozen/best_m3_frozen.pt`
   - **SHA256**: `5b22607444f4a242a52d0d9337e60c4c63044542dc6796a4a9de78c5ef38057c`
   - **Status**: **VERIFIED FROZEN PRIMARY**
2. **M3-Time+Delta (No-Mask)**:
   - **Path**: `experiments/m3_ablation_checkpoints/m3_ablation_no_mask.pt`
   - **Status**: **VERIFIED COMPONENT ABLATION**
3. **M3-Time+Mask (No-Time)**:
   - **Path**: `experiments/m3_ablation_checkpoints/m3_ablation_no_time.pt`
   - **Status**: **VERIFIED COMPONENT ABLATION**
4. **M2 / Values-Only (No-Time-No-Mask)**:
   - **Path**: `experiments/m3_ablation_checkpoints/m3_ablation_no_time_no_mask.pt`
   - **Status**: **VERIFIED MINIMAL BASELINE**
