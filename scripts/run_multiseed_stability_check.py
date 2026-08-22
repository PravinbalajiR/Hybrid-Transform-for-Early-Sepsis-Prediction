"""
run_multiseed_stability_check.py
---------------------------------
Multi-Seed Stability Analysis for M3 (Time-Aware Transformer).
Trains 5 NEW seeds ({1, 2, 3, 4, 5}) in addition to the original seed 42.
Evaluates all 6 checkpoints identically on validation and held-out test data.

Exports:
  - experiments/multiseed/seed_{N}/model.pt
  - experiments/multiseed/seed_{N}/test_predictions.npz
  - experiments/multiseed/seed_{N}/val_predictions.npz
  - results/multiseed/multiseed_summary.csv
"""

import sys
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import hashlib
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from utils.seed import set_seed
from preprocessing.load_data import load_dataset, ALL_FEATURE_COLS, LABEL_COL, DEFAULT_SET_A, DEFAULT_SET_B
from preprocessing.split import load_splits, make_splits, save_splits
from preprocessing.normalize import Normalizer
from preprocessing.dataset import create_dataloader
from models.transformer.tact_model import TACTModel as SepsisTransformer
from evaluation.utility_score import compute_utility_score
from scripts.run_m3_phase15_frozen_score_diagnostics import evaluate_policy_fast
from scripts.oracle_reconciliation_independent import calculate_best_single_alarm

EXP_DIR = BASE_DIR / "experiments" / "multiseed"
RESULTS_DIR = BASE_DIR / "results" / "multiseed"
EXP_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def print_flush(msg: str):
    print(msg, flush=True)

def compute_sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def compute_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i+1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    return float(ece)

def train_and_eval_seed(seed: int, dfs_train, dfs_val, dfs_test, device):
    set_seed(seed)
    input_dim = 34 * 3  # time_aware triplet input

    train_loader = create_dataloader(dfs_train, batch_size=32, shuffle=True)
    val_loader = create_dataloader(dfs_val, batch_size=32, shuffle=False)
    test_loader = create_dataloader(dfs_test, batch_size=32, shuffle=False)

    model = SepsisTransformer(input_dim=input_dim, d_model=64, nhead=4, num_layers=3).to(device)

    all_train_labels = np.concatenate([df[LABEL_COL].values for df in dfs_train])
    n_neg, n_pos = (all_train_labels == 0).sum(), (all_train_labels == 1).sum()
    pos_weight = torch.tensor([float(n_neg / n_pos)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    seed_dir = EXP_DIR / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = seed_dir / "model.pt"

    best_val_auc = -1.0
    patience = 5
    no_improve = 0

    for epoch in range(1, 15):
        model.train()
        for batch in train_loader:
            x = batch["triplet"].to(device)
            y = batch["label"].to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        # Val eval
        model.eval()
        val_probs_list, val_labels_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["triplet"].to(device)
                out = model(x)
                prob = torch.sigmoid(out).cpu().numpy()
                val_probs_list.append(prob)
                val_labels_list.append(batch["label"].numpy())

        val_probs_flat = np.concatenate(val_probs_list)
        val_labels_flat = np.concatenate(val_labels_list)
        val_auc = roc_auc_score(val_labels_flat, val_probs_flat)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {"seed": seed, "d_model": 64, "num_layers": 3, "nhead": 4},
                "epoch": epoch,
                "val_auroc": val_auc
            }, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # Load best weights
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()

    # Predict on Validation set
    val_probs_pat, val_labels_pat = [], []
    with torch.no_grad():
        for batch in val_loader:
            x = batch["triplet"].to(device)
            out = model(x)
            prob = torch.sigmoid(out).cpu().numpy()
            curr = 0
            for l in batch["lengths"]:
                val_probs_pat.append(prob[curr:curr+l])
                val_labels_pat.append(batch["label"].numpy()[curr:curr+l])
                curr += l

    # Find optimal threshold on validation set
    best_val_th = 0.19
    best_val_u = -999.0
    for th_c in np.arange(0.05, 0.95, 0.01):
        r_v = evaluate_policy_fast(val_probs_pat, val_labels_pat, threshold=float(th_c), cooldown_hours=36, policy_type="cooldown")
        if r_v["utility"] > best_val_u:
            best_val_u = r_v["utility"]
            best_val_th = float(th_c)

    # Predict on Test set
    test_probs_pat, test_labels_pat = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["triplet"].to(device)
            out = model(x)
            prob = torch.sigmoid(out).cpu().numpy()
            curr = 0
            for l in batch["lengths"]:
                test_probs_pat.append(prob[curr:curr+l])
                test_labels_pat.append(batch["label"].numpy()[curr:curr+l])
                curr += l

    test_y_true = np.concatenate(test_labels_pat)
    test_y_prob = np.concatenate(test_probs_pat)
    test_lens = [len(l) for l in test_labels_pat]

    # Save test predictions NPZ
    np.savez_compressed(
        seed_dir / "test_predictions.npz",
        y_true_flat=test_y_true,
        y_proba_flat=test_y_prob,
        patient_lengths=np.array(test_lens)
    )

    return ckpt_path, best_val_th, test_labels_pat, test_probs_pat, test_y_true, test_y_prob

def main():
    print_flush("=" * 95)
    print_flush("   MULTI-SEED STABILITY ANALYSIS FOR M3 TRANSFORMER")
    print_flush("=" * 95)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_flush(f"Using compute device: {device}")

    # Load dataset
    print_flush("Loading dataset files...")
    patient_dfs, _ = load_dataset(set_a_dir=DEFAULT_SET_A, set_b_dir=DEFAULT_SET_B, verbose=False)
    splits_dir = BASE_DIR / "data" / "splits"
    train_ids, val_ids, test_ids = load_splits(splits_dir)

    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)
    dfs_train = [df for df in patient_dfs if df["PatientID"].iloc[0] in train_set]
    dfs_val = [df for df in patient_dfs if df["PatientID"].iloc[0] in val_set]
    dfs_test = [df for df in patient_dfs if df["PatientID"].iloc[0] in test_set]

    # Normalize
    norm = Normalizer()
    norm.fit(dfs_train, ALL_FEATURE_COLS, verbose=False)
    dfs_train = norm.transform(dfs_train)
    dfs_val = norm.transform(dfs_val)
    dfs_test = norm.transform(dfs_test)

    # 1. Evaluate Original Frozen Checkpoint (Seed 42)
    print_flush("\n[SEED 42 - Original Frozen Checkpoint]")
    orig_ckpt_path = BASE_DIR / "experiments" / "final_m3_frozen" / "best_m3_frozen.pt"
    orig_hash = compute_sha256(orig_ckpt_path)
    print_flush(f"  Original Checkpoint SHA256: {orig_hash}")

    data_orig = np.load(BASE_DIR / "results" / "m3_final_test_predictions.npz", allow_pickle=True)
    orig_y_true, orig_y_prob, orig_lens = data_orig["y_true_flat"], data_orig["y_proba_flat"], data_orig["patient_lengths"]

    curr = 0
    orig_test_labels, orig_test_probs = [], []
    for l in orig_lens:
        orig_test_labels.append(orig_y_true[curr : curr + l])
        orig_test_probs.append(orig_y_prob[curr : curr + l])
        curr += l

    # Evaluate Original Seed
    orig_auroc = roc_auc_score(orig_y_true, orig_y_prob)
    orig_auprc = average_precision_score(orig_y_true, orig_y_prob)
    orig_brier = brier_score_loss(orig_y_true, orig_y_prob)
    orig_ece = compute_ece(orig_y_true, orig_y_prob)

    # Original Frozen Utility (th=0.190, C=36h)
    r_orig_frozen = evaluate_policy_fast(orig_test_probs, orig_test_labels, threshold=0.190, cooldown_hours=36, policy_type="cooldown")
    orig_frozen_u = r_orig_frozen["utility"]

    # Original Hindsight Grid Ceiling (C=72h)
    best_orig_grid_u = -999.0
    for th_c in np.arange(0.005, 0.995, 0.005):
        r_g = evaluate_policy_fast(orig_test_probs, orig_test_labels, threshold=float(th_c), cooldown_hours=72, policy_type="cooldown")
        if r_g["utility"] > best_orig_grid_u:
            best_orig_grid_u = r_g["utility"]

    # Ground Truth Oracle
    tot_ach_gt, tot_best_gt = 0.0, 0.0
    for lbls in orig_test_labels:
        _, a_g, b_g = calculate_best_single_alarm(lbls)
        tot_ach_gt += a_g
        tot_best_gt += b_g
    gt_oracle_u = tot_ach_gt / tot_best_gt

    seed_results = [{
        "Seed": 42,
        "Is_Original": True,
        "SHA256": orig_hash,
        "AUROC": orig_auroc,
        "AUPRC": orig_auprc,
        "Brier": orig_brier,
        "ECE": orig_ece,
        "Val_Selected_Thresh": 0.190,
        "FROZEN_MODEL_UTILITY": orig_frozen_u,
        "HINDSIGHT_GRID_CEILING": best_orig_grid_u,
        "GROUND_TRUTH_ORACLE": gt_oracle_u
    }]

    # 2. Train 5 NEW Seeds: {1, 2, 3, 4, 5}
    new_seeds = [1, 2, 3, 4, 5]
    all_hashes = [orig_hash]

    for sd in new_seeds:
        print_flush(f"\n[SEED {sd} - Retraining M3 Transformer...]")
        ckpt_p, val_th, test_lbls, test_prs, y_t, y_p = train_and_eval_seed(sd, dfs_train, dfs_val, dfs_test, device)
        sd_hash = compute_sha256(ckpt_p)
        all_hashes.append(sd_hash)
        print_flush(f"  Saved Checkpoint SHA256: {sd_hash}")

        sd_auroc = roc_auc_score(y_t, y_p)
        sd_auprc = average_precision_score(y_t, y_p)
        sd_brier = brier_score_loss(y_t, y_p)
        sd_ece = compute_ece(y_t, y_p)

        # Frozen Model Utility at this seed's validation-selected threshold
        r_sd_frozen = evaluate_policy_fast(test_prs, test_lbls, threshold=val_th, cooldown_hours=36, policy_type="cooldown")
        sd_frozen_u = r_sd_frozen["utility"]

        # Hindsight Grid Ceiling (C=72h)
        best_sd_grid_u = -999.0
        for th_c in np.arange(0.005, 0.995, 0.005):
            r_g = evaluate_policy_fast(test_prs, test_lbls, threshold=float(th_c), cooldown_hours=72, policy_type="cooldown")
            if r_g["utility"] > best_sd_grid_u:
                best_sd_grid_u = r_g["utility"]

        # GT Oracle verification
        tot_ach_s, tot_best_s = 0.0, 0.0
        for lbls in test_lbls:
            _, a_g, b_g = calculate_best_single_alarm(lbls)
            tot_ach_s += a_g
            tot_best_s += b_g
        sd_gt_u = tot_ach_s / tot_best_s

        seed_results.append({
            "Seed": sd,
            "Is_Original": False,
            "SHA256": sd_hash,
            "AUROC": sd_auroc,
            "AUPRC": sd_auprc,
            "Brier": sd_brier,
            "ECE": sd_ece,
            "Val_Selected_Thresh": val_th,
            "FROZEN_MODEL_UTILITY": sd_frozen_u,
            "HINDSIGHT_GRID_CEILING": best_sd_grid_u,
            "GROUND_TRUTH_ORACLE": sd_gt_u
        })

    df_multiseed = pd.DataFrame(seed_results)
    df_multiseed.to_csv(RESULTS_DIR / "multiseed_summary.csv", index=False)
    print_flush("\n" + df_multiseed.to_string(index=False))

    # 3. Variance Analysis across 6 Seeds
    distinct_hashes_pass = len(set(all_hashes)) == len(all_hashes)

    auroc_vals = df_multiseed["AUROC"].values
    auprc_vals = df_multiseed["AUPRC"].values
    frozen_u_vals = df_multiseed["FROZEN_MODEL_UTILITY"].values
    grid_u_vals = df_multiseed["HINDSIGHT_GRID_CEILING"].values
    gt_u_vals = df_multiseed["GROUND_TRUTH_ORACLE"].values

    gt_identical_pass = np.allclose(gt_u_vals, gt_oracle_u, atol=1e-6)

    # Outlier check for original seed
    orig_auroc_dist = abs(orig_auroc - np.mean(auroc_vals)) / np.std(auroc_vals)
    orig_frozen_dist = abs(orig_frozen_u - np.mean(frozen_u_vals)) / np.std(frozen_u_vals)
    is_outlier = (orig_auroc_dist > 1.0) or (orig_frozen_dist > 1.0)

    any_pos_frozen = any(u > 0.0 for u in frozen_u_vals)
    any_pos_grid = any(u > 0.0 for u in grid_u_vals)

    # 4. Mandatory Decision Gate Summary Output
    print_flush("\n" + "=" * 95)
    print_flush("MULTI-SEED STABILITY CHECK — AWAITING HUMAN REVIEW")
    print_flush("=" * 95)
    print_flush(f"Original frozen checkpoint seed        : 42")
    print_flush(f"New seeds tested                        : [1, 2, 3, 4, 5]")
    print_flush(f"All 6 checkpoint hashes distinct?       : [{'PASS' if distinct_hashes_pass else 'FAIL'}] ({len(set(all_hashes))}/6 unique)")
    print_flush(f"\nAUROC across 6 seeds      : mean={np.mean(auroc_vals):.4f}, std={np.std(auroc_vals):.4f}, min={np.min(auroc_vals):.4f}, max={np.max(auroc_vals):.4f}")
    print_flush(f"AUPRC across 6 seeds      : mean={np.mean(auprc_vals):.4f}, std={np.std(auprc_vals):.4f}, min={np.min(auprc_vals):.4f}, max={np.max(auprc_vals):.4f}")
    print_flush(f"FROZEN_MODEL_UTILITY       : mean={np.mean(frozen_u_vals):+.6f}, std={np.std(frozen_u_vals):.6f}, min={np.min(frozen_u_vals):+.6f}, max={np.max(frozen_u_vals):+.6f}")
    print_flush(f"HINDSIGHT_GRID_CEILING      : mean={np.mean(grid_u_vals):+.6f}, std={np.std(grid_u_vals):.6f}, min={np.min(grid_u_vals):+.6f}, max={np.max(grid_u_vals):+.6f}")
    print_flush(f"GROUND_TRUTH_ORACLE (should be identical): [{'CONFIRMED IDENTICAL' if gt_identical_pass else 'DIFFERS - INVESTIGATE'}] ({gt_oracle_u:+.6f})")
    print_flush(f"\nOriginal checkpoint an outlier vs 6-seed distribution? : [{'YES' if is_outlier else 'NO'}]")
    print_flush(f"Any seed achieving positive FROZEN_MODEL_UTILITY?       : [{'YES' if any_pos_frozen else 'NO'}]")
    print_flush(f"Any seed achieving positive HINDSIGHT_GRID_CEILING?     : [{'YES' if any_pos_grid else 'NO'}]")

    print_flush("\nCONCLUSION (present, do not auto-finalize):")
    if not any_pos_frozen and not any_pos_grid and np.std(frozen_u_vals) < 0.05:
        print_flush("  -> CASE B classification is ROBUST across training runs, not an")
        print_flush("     artifact of one lucky/unlucky seed.")
    else:
        print_flush("  -> CASE B classification is SEED-DEPENDENT and this must be")
        print_flush("     reported as a limitation.")
    print_flush("=" * 95)

if __name__ == "__main__":
    main()
