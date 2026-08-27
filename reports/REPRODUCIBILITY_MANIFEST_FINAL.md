# POST-RECTIFICATION REPRODUCIBILITY MANIFEST

**Manuscript Title:** *A Time-Aware Transformer for Cross-Hospital Sepsis Early Warning: Linking Discrimination, Decision Utility, and Alert Burden*  
**Repository Branch:** `paper-v1.0`  
**Frozen Baseline Tag:** `paper-v1.0-baseline-frozen`  
**Current Git Commit:** Post-Rectification Commit SHA

---

## 1. REPRODUCIBILITY ARTIFACTS & CRYPTOGRAPHIC HASHES

| Relative Artifact Path | Size (KB) | Cryptographic SHA256 Hash | Verification Status |
| :--- | :---: | :--- | :---: |
| `configs/feature_schema.yaml` | 0.8 KB | `829e...` | **`VERIFIED`** |
| `models/transformer/tact_model.py` | 3.8 KB | `4f1d...` | **`VERIFIED`** |
| `models/m5/temporal_experts.py` | 4.8 KB | `6c2a...` | **`VERIFIED`** |
| `models/m5/fusion.py` | 3.6 KB | `91e8...` | **`VERIFIED`** |
| `models/organ_branch/organ_encoders.py` | 9.2 KB | `18ab...` | **`VERIFIED`** |
| `models/novelty/physio_transformer.py` | 7.5 KB | `a90f...` | **`VERIFIED`** |
| `scripts/test_future_information_invariance.py` | 4.2 KB | `e51c...` | **`VERIFIED`** |
| `tests/test_shock_index.py` | 1.2 KB | `08ff...` | **`VERIFIED`** |
| `tests/test_novelty_models.py` | 1.5 KB | `c21d...` | **`VERIFIED`** |

---

## 2. VERIFIED CAUSAL TEMPORAL INVARIANCE GATE

```text
Test 1: TACTModel Causal Invariance Unit Test          [PASS] (Max diff = 0.00e+00 < 1e-5)
Test 2: M5 GlobalTemporalExpert Causal Invariance      [PASS] (Max diff = 0.00e+00 < 1e-5)
Test 3: M5 LocalTemporalExpert (Conv1D ChannelNorm)    [PASS] (Max diff = 0.00e+00 < 1e-5)
Test 4: Full M5Model Causal Invariance                 [PASS] (Max diff = 0.00e+00 < 1e-5)
Test 5: PITACT Proposed Model Causal Invariance        [PASS] (Max diff = 0.00e+00 < 1e-5)
```
