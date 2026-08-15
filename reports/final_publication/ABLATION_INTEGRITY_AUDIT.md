# Ablation Integrity & Leakage Audit Report

**Audit Date:** 2026-08-15  
**Cohort:** PhysioNet 2019 ICU Sepsis Challenge ($N=40,336$)  

---

## 1. Integrity Matrix

| Ablation Variant | Dataset Same | Split Same | Preprocessing Same | Labels Same | Normalization Same | Threshold Protocol | Audit Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **M3-Full** | PASS | PASS | PASS | PASS | PASS | Validation Only (th=0.60) | **PASS** |
| **M3-Time+Mask (No-Time)** | PASS | PASS | PASS | PASS | PASS | Validation Only (th=0.60) | **PASS** |
| **M3-Time+Delta (No-Mask)** | PASS | PASS | PASS | PASS | PASS | Validation Only (th=0.60) | **PASS** |
| **M2 / Values-Only** | PASS | PASS | PASS | PASS | PASS | Validation Only (th=0.60) | **PASS** |

---

## 2. Key Safeguards Verified
- **Patient Isolation**: 0 patient overlap across Train (18,302), Val (2,034), and Test (20,000) splits.
- **Normalization Isolation**: Z-score normalizer fit strictly on Training split.
- **Threshold Isolation**: Operating thresholds locked on Validation split ONLY; Test set evaluated single-pass.
