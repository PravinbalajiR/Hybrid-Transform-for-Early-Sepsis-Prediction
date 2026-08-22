# 🔒 RESEARCH FREEZE MANIFEST (TASK 13)

**Project:** `Hybrid-Transform-for-Early-Sepsis-Prediction`  
**Branch:** `paper-v1.0`  
**Freeze Status:** **PERMANENTLY FROZEN — NO FURTHER MODEL RETRAINING OR POLICY SEARCH**

---

## 1. Frozen Codebase Elements

The following project components are permanently frozen and must NOT be altered under any circumstances:

1. **Dataset & Patient Splits:**
   - Set A (Emory): Train split ($N = 16,192$) and Validation split ($N = 4,144$).
   - Set B (BIDMC): Held-out Test cohort ($N = 20,000$, $1,066$ septic, $18,934$ non-septic).
2. **Frozen Base Checkpoint:**
   - Path: `experiments/final_m3_frozen/best_m3_frozen.pt`
   - SHA256: `5b22607444f4a242a52d0d9337e60c4c63044542dc6796a4a9de78c5ef38057c`
3. **Official PhysioNet 2019 Utility Metric:**
   - Reward: $+1.0$ (optimal TP alarm at $\max(0, t_{\text{onset}}-6\text{h})$).
   - Early/Late Penalties: Linear ramp from $0.0$ at $t_{\text{early}}=12\text{h}$ pre-onset to $+1.0$ at $t_{\text{optimal}}=6\text{h}$ pre-onset; linear decay to $0.0$ at $t_{\text{late}}=3\text{h}$ post-onset.
   - False Alarm Penalty: $-0.05$ points per hour.
   - Missed Sepsis Penalty: $-2.0$ points per patient.
4. **Prespecified Deployable Policy:**
   - Validation-selected threshold: $th_{\text{val}}^* = 0.190$.
   - Cooldown alert suppression: $C = 36\text{h}$.
5. **Authoritative Taxonomy:**
   - `GROUND_TRUTH_ORACLE_CEILING` (`+0.826246`)
   - `FROZEN_MODEL_UTILITY` (`-0.257312`)
   - `HINDSIGHT_GRID_SCORE_POLICY_CEILING` (`-0.198307`)
   - `PATIENT_ADAPTIVE_THRESHOLD_CEILING` (`+0.281895`, Counterfactual Diagnostic Only)
   - `REALISTIC_ACHIEVABLE_UTILITY` (`-0.198307`)

---

## 2. Forbidden Actions

The following actions are strictly prohibited:
- Retraining any neural network architecture (Transformers, DANNs, CNNs, GRUs).
- Searching for new hyperparameter or loss function configurations.
- Tuning decision thresholds using held-out BIDMC test set outcomes.
- Modifying test label arrays or patient split JSON files.
- Introducing new "oracle" definitions or renaming metrics for reporting bias.
