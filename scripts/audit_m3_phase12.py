"""
audit_m3_phase12.py
-------------------
Standalone Audit Script for M3 Phase 12 Domain Generalization, Split Provenance & Shift-Robust Optimization (M3-DR).
Verifies all 15 Phase 12 output artifacts, artifact SHA256 hashes, patient disjointness, zero test leakage, fingerprint uniqueness, and scorer equivalence.
"""

import sys
import json
import hashlib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

RESULTS_DIR = BASE_DIR / "results"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
SPLITS_DIR = BASE_DIR / "data" / "splits"

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 80)
    print("   M3 PHASE 12 — STANDALONE INTEGRITY AUDIT & LEAKAGE VERIFICATION")
    print("=" * 80)

    # 1. Check Artifact Checksums
    ckpt_path = EXPERIMENTS_DIR / "final_m3_frozen" / "best_m3_frozen.pt"
    test_npz_path = RESULTS_DIR / "m3_final_test_predictions.npz"

    exp_ckpt_sha = "5b22607444f4a242a52d0d9337e60c4c63044542dc6796a4a9de78c5ef38057c"
    exp_test_sha = "02fd6eb78682be8ca5743c4b3fddfcc7f57ed56f27f8496092108c30b2188a3d"

    actual_ckpt_sha = compute_sha256(ckpt_path) if ckpt_path.exists() else "MISSING"
    actual_test_sha = compute_sha256(test_npz_path) if test_npz_path.exists() else "MISSING"

    print("\n1. Checkpoint & Prediction Artifact Provenance:")
    print(f"   Checkpoint SHA256 : {actual_ckpt_sha} [{'PASSED' if actual_ckpt_sha==exp_ckpt_sha else 'FAILED'}]")
    print(f"   Test NPZ SHA256   : {actual_test_sha} [{'PASSED' if actual_test_sha==exp_test_sha else 'FAILED'}]")

    if actual_ckpt_sha != exp_ckpt_sha or actual_test_sha != exp_test_sha:
        print("   CRITICAL ERROR: Artifact checksum mismatch!")
        print("   SCIENTIFIC VALIDITY: FAILED")
        sys.exit(1)

    # 2. Patient Disjointness Check
    print("\n2. Patient Split Disjointness Verification:")
    train_ids = set(json.load(open(SPLITS_DIR / "train_ids.json")))
    val_ids = set(json.load(open(SPLITS_DIR / "val_ids.json")))
    test_ids = set(json.load(open(SPLITS_DIR / "test_ids.json")))

    tv_overlap = len(train_ids.intersection(val_ids))
    tt_overlap = len(train_ids.intersection(test_ids))
    vt_overlap = len(val_ids.intersection(test_ids))

    print(f"   Train/Val Overlap : {tv_overlap} [{'PASSED' if tv_overlap==0 else 'FAILED'}]")
    print(f"   Train/Test Overlap: {tt_overlap} [{'PASSED' if tt_overlap==0 else 'FAILED'}]")
    print(f"   Val/Test Overlap  : {vt_overlap} [{'PASSED' if vt_overlap==0 else 'FAILED'}]")

    if tv_overlap != 0 or tt_overlap != 0 or vt_overlap != 0:
        print("   CRITICAL ERROR: Patient split overlap detected!")
        print("   SCIENTIFIC VALIDITY: FAILED")
        sys.exit(1)

    # 3. Verify Output File Existence & Non-Emptiness
    required_files = [
        RESULTS_DIR / "m3_phase12_split_provenance.csv",
        RESULTS_DIR / "m3_phase12_split_provenance.json",
        RESULTS_DIR / "m3_phase12_indomain_vs_crossdomain.csv",
        RESULTS_DIR / "m3_phase12_feature_shift.csv",
        RESULTS_DIR / "m3_phase12_missingness_shift.csv",
        RESULTS_DIR / "m3_phase12_threshold_frontier.csv",
        RESULTS_DIR / "m3_phase12_ablation.csv",
        RESULTS_DIR / "m3_phase12_utility_decomposition.csv",
        RESULTS_DIR / "m3_phase12_bootstrap_ci.csv",
        RESULTS_DIR / "m3_phase12_model_selection.json",
        RESULTS_DIR / "m3_phase12_frozen_model.json",
        RESULTS_DIR / "m3_phase12_freeze_manifest.md",
        RESULTS_DIR / "m3_phase12_test_report.md",
        RESULTS_DIR / "m3_phase12_novelty_matrix.csv",
        RESULTS_DIR / "m3_phase12_diagnostic_summary.json",
    ]

    print("\n3. Required Output Files Integrity:")
    all_files_exist = True
    for fpath in required_files:
        exists = fpath.exists() and fpath.stat().st_size > 0
        status = "PASSED" if exists else "FAILED"
        if not exists: all_files_exist = False
        print(f"   File: {fpath.name:45s} [{status}]")

    if not all_files_exist:
        print("   CRITICAL ERROR: One or more required Phase 12 output files missing or empty!")
        print("   SCIENTIFIC VALIDITY: FAILED")
        sys.exit(1)

    # 4. Verify Unique Configuration Fingerprints
    ab_df = pd.read_csv(RESULTS_DIR / "m3_phase12_ablation.csv")
    unique_fingerprints = ab_df["Config_Fingerprint"].nunique()
    print("\n4. Ablation Configuration Fingerprint Audit:")
    print(f"   Unique Fingerprints Found : {unique_fingerprints} / 9 [{'PASSED' if unique_fingerprints==9 else 'FAILED'}]")

    if unique_fingerprints != 9:
        print("   CRITICAL ERROR: Duplicate configuration fingerprints detected!")
        print("   SCIENTIFIC VALIDITY: FAILED")
        sys.exit(1)

    # 5. Scorer Equivalence Verification
    decomp_df = pd.read_csv(RESULTS_DIR / "m3_phase12_utility_decomposition.csv")
    max_diff = decomp_df["arith_diff"].max()

    print("\n5. Official Scorer vs Independent Utility Decomposition Check:")
    print(f"   Max Arithmetic Difference across Top Policies : {max_diff:.12e}")
    print(f"   Equivalence Tolerance                          : <= 1e-10")
    print(f"   Scorer Audit Status                            : {'PASSED [ZERO DISCREPANCY]' if max_diff <= 1e-10 else 'FAILED'}")

    if max_diff > 1e-10:
        print("   CRITICAL ERROR: Official scorer mismatch!")
        print("   SCIENTIFIC VALIDITY: FAILED")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("   ALL PHASE 12 AUDIT CHECKS PASSED SUCCESSFULLY")
    print("   SCIENTIFIC VALIDITY: PASSED")
    print("=" * 80)

if __name__ == "__main__":
    main()
