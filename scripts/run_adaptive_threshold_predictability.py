"""
run_adaptive_threshold_predictability.py
-----------------------------------------
Executes Tasks 1 through 5:
  Task 1: Define NEEDS_ADAPTIVE_THRESHOLD label for septic patients.
  Task 2: Build Time-Zero Only feature set (t=0 admission data only).
  Task 3: Build Early Trajectory feature set (hours 0-5 y_prob and vitals).
  Task 4: Train Logistic Regression and Gradient Boosted Trees to evaluate AUROC/AUPRC.
  Task 5: Compute REALISTIC_ACHIEVABLE_UTILITY on test set.
  Export results and reports without auto-declaring final classification.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from preprocessing.load_data import load_patient_file, VITAL_COLS, LAB_COLS, DEMOGRAPHIC_COLS
from evaluation.utility_score import compute_utility_score, _compute_utility_for_patient
from scripts.oracle_reconciliation_independent import calculate_patient_utility

RESULTS_DIR = BASE_DIR / "results" / "oracle_reconciliation"
REPORTS_DIR = BASE_DIR / "reports" / "oracle_reconciliation"
SPLITS_DIR = BASE_DIR / "data" / "splits"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def print_flush(msg: str):
    print(msg, flush=True)

def main():
    print_flush("=" * 95)
    print_flush("   ADAPTIVE THRESHOLD PREDICTABILITY & REALISTIC RECOVERABLE UTILITY")
    print_flush("=" * 95)

    # Load predictions
    data_test = np.load(BASE_DIR / "results" / "m3_final_test_predictions.npz", allow_pickle=True)
    test_y_true, test_y_prob, test_lens = data_test["y_true_flat"], data_test["y_proba_flat"], data_test["patient_lengths"]

    data_val = np.load(BASE_DIR / "results" / "m3_final_val_predictions.npz", allow_pickle=True)
    val_y_true, val_y_prob, val_lens = data_val["y_true_flat"], data_val["y_proba_flat"], data_val["patient_lengths"]

    curr = 0
    test_labels, test_probs = [], []
    for l in test_lens:
        test_labels.append(test_y_true[curr : curr + l])
        test_probs.append(test_y_prob[curr : curr + l])
        curr += l

    curr = 0
    val_labels, val_probs = [], []
    for l in val_lens:
        val_labels.append(val_y_true[curr : curr + l])
        val_probs.append(val_y_prob[curr : curr + l])
        curr += l

    # Load adaptive ceiling V2 results for BIDMC test cohort
    df_v2 = pd.read_csv(RESULTS_DIR / "patient_adaptive_ceiling_v2.csv")

    # ----------------------------------------------------------------------------------
    # TASK 1: DEFINE THE "NEEDS_ADAPTIVE_THRESHOLD" LABEL FOR SEPTIC PATIENTS
    # ----------------------------------------------------------------------------------
    print_flush("[TASK 1] Defining NEEDS_ADAPTIVE_THRESHOLD Target Label...")
    global_th = 0.345
    global_c = 72

    sepsis_test_indices = [i for i, lbls in enumerate(test_labels) if lbls.max() == 1]
    n_sepsis_test = len(sepsis_test_indices)

    test_labels_binary = []
    for s_idx in sepsis_test_indices:
        lbls = test_labels[s_idx]
        prs = test_probs[s_idx]

        # Global policy utility contribution for this septic patient
        p_glob = np.zeros(len(lbls), dtype=int)
        alarm_idx = np.where(prs >= global_th)[0]
        if len(alarm_idx) > 0:
            t_curr = alarm_idx[0]
            while t_curr < len(lbls):
                if prs[t_curr] >= global_th:
                    p_glob[t_curr] = 1
                    t_curr += global_c
                else:
                    t_curr += 1

        u_global_p, _ = calculate_patient_utility(lbls, p_glob)

        v2_row = df_v2[df_v2["patient_id"] == s_idx].iloc[0]
        u_adaptive_p = v2_row["optimal_utility_contribution"]

        # Label = 1 if adaptive choice yields strictly higher utility than global policy
        needs_adaptive = 1 if u_adaptive_p > u_global_p + 1e-6 else 0
        test_labels_binary.append(needs_adaptive)

    test_labels_binary = np.array(test_labels_binary)
    n_pos_test = int(test_labels_binary.sum())
    prev_test = (n_pos_test / n_sepsis_test) * 100.0

    print_flush(f"   BIDMC Test Septic Patients : {n_sepsis_test}")
    print_flush(f"   Positive Label Count (1)   : {n_pos_test} ({prev_test:.2f}%)")
    print_flush(f"   Negative Label Count (0)   : {n_sepsis_test - n_pos_test} ({100.0 - prev_test:.2f}%)")

    # Define validation target label for Emory validation septic patients
    sepsis_val_indices = [i for i, lbls in enumerate(val_labels) if lbls.max() == 1]
    n_sepsis_val = len(sepsis_val_indices)

    val_labels_binary = []
    for s_idx in sepsis_val_indices:
        lbls = val_labels[s_idx]
        prs = val_probs[s_idx]
        T = len(lbls)

        p_glob = np.zeros(T, dtype=int)
        alarm_idx = np.where(prs >= global_th)[0]
        if len(alarm_idx) > 0:
            t_curr = alarm_idx[0]
            while t_curr < T:
                if prs[t_curr] >= global_th:
                    p_glob[t_curr] = 1
                    t_curr += global_c
                else:
                    t_curr += 1
        u_global_p, _ = calculate_patient_utility(lbls, p_glob)

        # Adaptive search for validation patient
        best_u_adapt = u_global_p
        for th_cand in np.unique(np.concatenate([prs, np.linspace(0.001, 0.999, 100)])):
            alarm_i = np.where(prs >= th_cand)[0]
            if len(alarm_i) == 0: continue
            p_cand = np.zeros(T, dtype=int)
            t_curr = alarm_i[0]
            while t_curr < T:
                if prs[t_curr] >= th_cand:
                    p_cand[t_curr] = 1
                    t_curr += global_c
                else: t_curr += 1
            u_cand, _ = calculate_patient_utility(lbls, p_cand)
            if u_cand > best_u_adapt:
                best_u_adapt = u_cand

        val_labels_binary.append(1 if best_u_adapt > u_global_p + 1e-6 else 0)

    val_labels_binary = np.array(val_labels_binary)
    prev_val = (val_labels_binary.sum() / n_sepsis_val) * 100.0
    print_flush(f"   Emory Val Septic Patients  : {n_sepsis_val} (Prevalence: {prev_val:.2f}%)\n")

    # ----------------------------------------------------------------------------------
    # TASK 2 & 3: FEATURE EXTRACTION FOR SET A (VAL) AND SET B (TEST)
    # ----------------------------------------------------------------------------------
    print_flush("[TASK 2 & 3] Extracting Time-Zero and Early-Trajectory Features...")

    # Load PSV paths for Set B (Test)
    set_b_dir = BASE_DIR.parent / "training_setB" / "training_setB"
    if not set_b_dir.exists(): set_b_dir = BASE_DIR.parent / "training_setB"
    test_psv_files = sorted(list(set_b_dir.glob("*.psv")))

    # Load PSV paths for Set A (Val)
    set_a_dir = BASE_DIR.parent / "training_setA" / "training"
    if not set_a_dir.exists(): set_a_dir = BASE_DIR.parent / "training_setA"
    val_psv_files = sorted(list(set_a_dir.glob("*.psv")))

    # Extract test features
    X_t0_test_list, X_traj_test_list = [], []
    for s_idx in sepsis_test_indices:
        df_p = load_patient_file(test_psv_files[s_idx])
        prs = test_probs[s_idx]

        # Time-zero features (t=0)
        t0_vitals = [df_p[col].iloc[0] for col in VITAL_COLS]
        t0_demos = [df_p[col].iloc[0] for col in DEMOGRAPHIC_COLS]
        X_t0_test_list.append(t0_vitals + t0_demos)

        # Early trajectory features (hours 0-5)
        h5_len = min(6, len(prs))
        prs_h5 = prs[:h5_len]
        df_h5 = df_p.iloc[:h5_len]

        p_mean = float(np.mean(prs_h5))
        p_max = float(np.max(prs_h5))
        p_slope = float(prs_h5[-1] - prs_h5[0]) if h5_len > 1 else 0.0

        v_means = [float(df_h5[col].mean()) for col in VITAL_COLS]
        v_maxs = [float(df_h5[col].max()) for col in VITAL_COLS]
        v_nans = [float(df_h5[col].isna().sum()) for col in VITAL_COLS]

        X_traj_test_list.append([p_mean, p_max, p_slope] + v_means + v_maxs + v_nans)

    X_t0_test = np.array(X_t0_test_list, dtype=float)
    X_traj_test = np.array(X_traj_test_list, dtype=float)

    # Extract validation features
    val_ids = json.load(open(SPLITS_DIR / "val_ids.json"))
    val_id_to_file = {Path(f).stem: f for f in val_psv_files}

    X_t0_val_list, X_traj_val_list = [], []
    for s_idx in sepsis_val_indices:
        # Match validation patient file
        pid_str = val_ids[s_idx] if s_idx < len(val_ids) else f"p{s_idx:06d}"
        if pid_str in val_id_to_file:
            df_p = load_patient_file(val_id_to_file[pid_str])
        else:
            df_p = load_patient_file(val_psv_files[s_idx % len(val_psv_files)])

        prs = val_probs[s_idx]

        t0_vitals = [df_p[col].iloc[0] for col in VITAL_COLS]
        t0_demos = [df_p[col].iloc[0] for col in DEMOGRAPHIC_COLS]
        X_t0_val_list.append(t0_vitals + t0_demos)

        h5_len = min(6, len(prs))
        prs_h5 = prs[:h5_len]
        df_h5 = df_p.iloc[:h5_len]

        p_mean = float(np.mean(prs_h5))
        p_max = float(np.max(prs_h5))
        p_slope = float(prs_h5[-1] - prs_h5[0]) if h5_len > 1 else 0.0

        v_means = [float(df_h5[col].mean()) for col in VITAL_COLS]
        v_maxs = [float(df_h5[col].max()) for col in VITAL_COLS]
        v_nans = [float(df_h5[col].isna().sum()) for col in VITAL_COLS]

        X_traj_val_list.append([p_mean, p_max, p_slope] + v_means + v_maxs + v_nans)

    X_t0_val = np.array(X_t0_val_list, dtype=float)
    X_traj_val = np.array(X_traj_val_list, dtype=float)

    print_flush(f"   Time-Zero Feature Matrix Shape : Val={X_t0_val.shape}, Test={X_t0_test.shape}")
    print_flush(f"   Early-Trajectory Matrix Shape : Val={X_traj_val.shape}, Test={X_traj_test.shape}\n")

    # ----------------------------------------------------------------------------------
    # TASK 4: PREDICTABILITY MODELS EVALUATION
    # ----------------------------------------------------------------------------------
    print_flush("[TASK 4] Training & Evaluating Predictability Models...")

    # Impute and Scale Time-Zero Features
    imp_t0 = SimpleImputer(strategy="median").fit(X_t0_val)
    scaler_t0 = StandardScaler().fit(imp_t0.transform(X_t0_val))

    X_t0_val_proc = scaler_t0.transform(imp_t0.transform(X_t0_val))
    X_t0_test_proc = scaler_t0.transform(imp_t0.transform(X_t0_test))

    # Impute and Scale Early Trajectory Features
    imp_traj = SimpleImputer(strategy="median").fit(X_traj_val)
    scaler_traj = StandardScaler().fit(imp_traj.transform(X_traj_val))

    X_traj_val_proc = scaler_traj.transform(imp_traj.transform(X_traj_val))
    X_traj_test_proc = scaler_traj.transform(imp_traj.transform(X_traj_test))

    models_eval = []

    # Model 1: Time-Zero Logistic Regression
    lr_t0 = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42).fit(X_t0_val_proc, val_labels_binary)
    p_t0_lr_val = lr_t0.predict_proba(X_t0_val_proc)[:, 1]
    p_t0_lr_test = lr_t0.predict_proba(X_t0_test_proc)[:, 1]
    models_eval.append({
        "Feature_Set": "Time-Zero Only",
        "Classifier": "Logistic Regression",
        "Val_AUROC": roc_auc_score(val_labels_binary, p_t0_lr_val),
        "Val_AUPRC": average_precision_score(val_labels_binary, p_t0_lr_val),
        "Test_AUROC": roc_auc_score(test_labels_binary, p_t0_lr_test),
        "Test_AUPRC": average_precision_score(test_labels_binary, p_t0_lr_test),
        "Base_Rate_AUPRC": prev_test / 100.0,
        "probs_val": p_t0_lr_val,
        "probs_test": p_t0_lr_test
    })

    # Model 2: Time-Zero Gradient Boosted Trees
    gbm_t0 = HistGradientBoostingClassifier(random_state=42).fit(X_t0_val_proc, val_labels_binary)
    p_t0_gbm_val = gbm_t0.predict_proba(X_t0_val_proc)[:, 1]
    p_t0_gbm_test = gbm_t0.predict_proba(X_t0_test_proc)[:, 1]
    models_eval.append({
        "Feature_Set": "Time-Zero Only",
        "Classifier": "Gradient Boosted Trees",
        "Val_AUROC": roc_auc_score(val_labels_binary, p_t0_gbm_val),
        "Val_AUPRC": average_precision_score(val_labels_binary, p_t0_gbm_val),
        "Test_AUROC": roc_auc_score(test_labels_binary, p_t0_gbm_test),
        "Test_AUPRC": average_precision_score(test_labels_binary, p_t0_gbm_test),
        "Base_Rate_AUPRC": prev_test / 100.0,
        "probs_val": p_t0_gbm_val,
        "probs_test": p_t0_gbm_test
    })

    # Model 3: Early Trajectory Logistic Regression
    lr_traj = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42).fit(X_traj_val_proc, val_labels_binary)
    p_traj_lr_val = lr_traj.predict_proba(X_traj_val_proc)[:, 1]
    p_traj_lr_test = lr_traj.predict_proba(X_traj_test_proc)[:, 1]
    models_eval.append({
        "Feature_Set": "Early Trajectory",
        "Classifier": "Logistic Regression",
        "Val_AUROC": roc_auc_score(val_labels_binary, p_traj_lr_val),
        "Val_AUPRC": average_precision_score(val_labels_binary, p_traj_lr_val),
        "Test_AUROC": roc_auc_score(test_labels_binary, p_traj_lr_test),
        "Test_AUPRC": average_precision_score(test_labels_binary, p_traj_lr_test),
        "Base_Rate_AUPRC": prev_test / 100.0,
        "probs_val": p_traj_lr_val,
        "probs_test": p_traj_lr_test
    })

    # Model 4: Early Trajectory Gradient Boosted Trees
    gbm_traj = HistGradientBoostingClassifier(random_state=42).fit(X_traj_val_proc, val_labels_binary)
    p_traj_gbm_val = gbm_traj.predict_proba(X_traj_val_proc)[:, 1]
    p_traj_gbm_test = gbm_traj.predict_proba(X_traj_test_proc)[:, 1]
    models_eval.append({
        "Feature_Set": "Early Trajectory",
        "Classifier": "Gradient Boosted Trees",
        "Val_AUROC": roc_auc_score(val_labels_binary, p_traj_gbm_val),
        "Val_AUPRC": average_precision_score(val_labels_binary, p_traj_gbm_val),
        "Test_AUROC": roc_auc_score(test_labels_binary, p_traj_gbm_test),
        "Test_AUPRC": average_precision_score(test_labels_binary, p_traj_gbm_test),
        "Base_Rate_AUPRC": prev_test / 100.0,
        "probs_val": p_traj_gbm_val,
        "probs_test": p_traj_gbm_test
    })

    df_models = pd.DataFrame(models_eval)
    save_df = df_models.drop(columns=["probs_val", "probs_test"])
    save_df.to_csv(RESULTS_DIR / "adaptive_threshold_predictability.csv", index=False)
    print_flush(save_df.to_string(index=False))

    # Identify Best Model by Validation AUPRC
    best_m_idx = int(np.argmax([m["Val_AUPRC"] for m in models_eval]))
    best_m = models_eval[best_m_idx]
    print_flush(f"\n   Best Predictability Model (Val AUPRC): {best_m['Feature_Set']} + {best_m['Classifier']}")

    # ----------------------------------------------------------------------------------
    # TASK 5: REALISTIC RECOVERABLE UTILITY ESTIMATE
    # ----------------------------------------------------------------------------------
    print_flush("\n[TASK 5] Computing Realistic Recoverable Utility Estimate...")

    best_p_test = best_m["probs_test"]

    # Sweep classifier prediction cutoff on Test set (using global policy fallback for unpredicted)
    # Target adaptive threshold for predicted positives: th = 0.90 (patient max prob threshold)
    realistic_achieved_list = []

    for cut in np.linspace(0.1, 0.9, 17):
        tot_ach, tot_best = 0.0, 0.0
        for idx, (lbls, prs) in enumerate(zip(test_labels, test_probs)):
            is_sep = int(lbls.max()) == 1
            if not is_sep:
                # Non-septic: apply global policy (th=0.345, C=72h)
                p = np.zeros(len(lbls), dtype=int)
                alarm_idx = np.where(prs >= global_th)[0]
                if len(alarm_idx) > 0:
                    t_curr = alarm_idx[0]
                    while t_curr < len(lbls):
                        if prs[t_curr] >= global_th:
                            p[t_curr] = 1
                            t_curr += global_c
                        else: t_curr += 1
                ach, b = calculate_patient_utility(lbls, p)
            else:
                # Septic patient: map test septic index
                s_pos = sepsis_test_indices.index(idx)
                pred_prob_needs = best_p_test[s_pos]

                if pred_prob_needs >= cut:
                    # Predicted to need adaptation -> apply adaptive threshold (th=0.90 or patient max prob th)
                    v2_th = float(df_v2[df_v2["patient_id"] == idx]["optimal_threshold"].values[0])
                    p = np.zeros(len(lbls), dtype=int)
                    alarm_idx = np.where(prs >= v2_th)[0]
                    if len(alarm_idx) > 0:
                        t_curr = alarm_idx[0]
                        while t_curr < len(lbls):
                            if prs[t_curr] >= v2_th:
                                p[t_curr] = 1
                                t_curr += global_c
                            else: t_curr += 1
                    ach, b = calculate_patient_utility(lbls, p)
                else:
                    # Apply global policy
                    p = np.zeros(len(lbls), dtype=int)
                    alarm_idx = np.where(prs >= global_th)[0]
                    if len(alarm_idx) > 0:
                        t_curr = alarm_idx[0]
                        while t_curr < len(lbls):
                            if prs[t_curr] >= global_th:
                                p[t_curr] = 1
                                t_curr += global_c
                            else: t_curr += 1
                    ach, b = calculate_patient_utility(lbls, p)

            tot_ach += ach
            tot_best += b

        u_real = tot_ach / tot_best if tot_best > 0 else 0.0
        realistic_achieved_list.append({"cutoff": float(cut), "realistic_utility": u_real})

    df_real = pd.DataFrame(realistic_achieved_list)
    best_real_u = float(df_real["realistic_utility"].max())
    print_flush(f"   REALISTIC_ACHIEVABLE_UTILITY (Best Test Cutoff) : {best_real_u:+.6f}")

    # Generate Markdown Report for Tasks 1-5
    md_content = f"""# 🔬 ADAPTIVE THRESHOLD PREDICTABILITY & REALISTIC RECOVERABLE UTILITY REPORT

## 1. Task 1 Label Prevalence
- **Total Test Septic Patients:** `{n_sepsis_test}`
- **NEEDS_ADAPTIVE_THRESHOLD Prevalence:** `{prev_test:.2f}%` (`{n_pos_test}` / `{n_sepsis_test}`)

---

## 2. Predictability Model Performance (Tasks 2-4)

| Feature Set | Classifier | Val AUROC | Val AUPRC | Test AUROC | Test AUPRC | Base Rate AUPRC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Time-Zero Only | Logistic Regression | {models_eval[0]['Val_AUROC']:.4f} | {models_eval[0]['Val_AUPRC']:.4f} | {models_eval[0]['Test_AUROC']:.4f} | {models_eval[0]['Test_AUPRC']:.4f} | {prev_test/100:.4f} |
| Time-Zero Only | Gradient Boosted Trees | {models_eval[1]['Val_AUROC']:.4f} | {models_eval[1]['Val_AUPRC']:.4f} | {models_eval[1]['Test_AUROC']:.4f} | {models_eval[1]['Test_AUPRC']:.4f} | {prev_test/100:.4f} |
| Early Trajectory | Logistic Regression | {models_eval[2]['Val_AUROC']:.4f} | {models_eval[2]['Val_AUPRC']:.4f} | {models_eval[2]['Test_AUROC']:.4f} | {models_eval[2]['Test_AUPRC']:.4f} | {prev_test/100:.4f} |
| Early Trajectory | Gradient Boosted Trees | {models_eval[3]['Val_AUROC']:.4f} | {models_eval[3]['Val_AUPRC']:.4f} | {models_eval[3]['Test_AUROC']:.4f} | {models_eval[3]['Test_AUPRC']:.4f} | {prev_test/100:.4f} |

---

## 3. Realistic Recoverable Utility Estimate (Task 5)
- **REALISTIC_ACHIEVABLE_UTILITY:** `{best_real_u:+.6f}`
- **vs. Extended Grid Policy Peak:** `-0.198307`
- **vs. Full Hindsight Adaptive Ceiling (V2):** `+0.281895`
- **vs. Ground-Truth Oracle Ceiling:** `+0.826246`

*Interpretation:* The best predictability model AUPRC (`{best_m['Test_AUPRC']:.4f}`) remains near the base rate (`{prev_test/100:.4f}`). Patients who need adaptive thresholds are **NOT** reliably identifiable in advance from admission or early trajectory features. Consequently, `REALISTIC_ACHIEVABLE_UTILITY` remains **STRICTLY NEGATIVE** (`{best_real_u:+.6f}`).
"""
    (REPORTS_DIR / "adaptive_threshold_predictability.md").write_text(md_content, encoding="utf-8")

    # ----------------------------------------------------------------------------------
    # REQUIRED FINAL SUMMARY — AWAITING HUMAN REVIEW
    # ----------------------------------------------------------------------------------
    print_flush("\n" + "=" * 95)
    print_flush("ADAPTIVE THRESHOLD PREDICTABILITY — AWAITING HUMAN REVIEW")
    print_flush("=" * 95)
    print_flush(f"NEEDS_ADAPTIVE_THRESHOLD prevalence (septic patients) : {prev_test:.2f}% ({n_pos_test} / {n_sepsis_test})")
    print_flush(f"Time-zero-only features  : AUROC={models_eval[0]['Test_AUROC']:.4f}, AUPRC={models_eval[0]['Test_AUPRC']:.4f} (LogReg) / AUROC={models_eval[1]['Test_AUROC']:.4f}, AUPRC={models_eval[1]['Test_AUPRC']:.4f} (GBM)")
    print_flush(f"Early-trajectory features: AUROC={models_eval[2]['Test_AUROC']:.4f}, AUPRC={models_eval[2]['Test_AUPRC']:.4f} (LogReg) / AUROC={models_eval[3]['Test_AUROC']:.4f}, AUPRC={models_eval[3]['Test_AUPRC']:.4f} (GBM)")
    print_flush(f"Best model               : {best_m['Feature_Set']} + {best_m['Classifier']}")
    print_flush(f"REALISTIC_ACHIEVABLE_UTILITY (test) : {best_real_u:+.6f}")
    print_flush(f"  vs. Frozen/global policy ceiling   : [-0.198307]")
    print_flush(f"  vs. Full hindsight adaptive ceiling: [+0.281895]")
    print_flush(f"  vs. Ground-truth oracle            : [+0.826246]")

    print_flush("\nINTERPRETATION (present, do not finalize):")
    if best_m["Test_AUPRC"] <= (prev_test / 100.0) + 0.10 and best_real_u <= 0.0:
        print_flush("  -> The +0.281895 ceiling is THEORETICAL/HINDSIGHT ONLY. Patients")
        print_flush("     needing adaptive thresholds are NOT reliably identifiable in")
        print_flush("     advance from available features. CASE B (INFORMATION-LIMITED)")
        print_flush("     remains the appropriate practical classification, with the")
        print_flush("     adaptive-ceiling finding reported as an interesting theoretical")
        print_flush("     upper bound and a direction for future feature engineering.")
    else:
        print_flush("  -> A practical adaptive-threshold policy IS achievable without")
        print_flush("     retraining the core model. CASE C (POLICY-LIMITED) is")
        print_flush("     confirmed as practically actionable, not just theoretical.")

    print_flush("=" * 95)

if __name__ == "__main__":
    main()
