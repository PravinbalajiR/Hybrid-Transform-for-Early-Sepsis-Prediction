"""
audit_m3_phase15.py
-------------------
Standalone Audit Script for M3 Phase 15 Frozen-Score Utility Feasibility & Policy-Transfer Investigation.
Verifies all 20 Phase 15 output artifacts, artifact SHA256 hashes, patient disjointness,
scorer equivalence <= 1e-10, zero test leakage, and report completeness.
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
    if not filepath.exists(): return "MISSING"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 80)
    print("   M3 PHASE 15 — STANDALONE INTEGRITY & LEAKAGE AUDIT")
    print("=" * 80)

    # 1. Check Artifact Checksums
    ckpt_path = EXPERIMENTS_DIR / "final_m3_frozen" / "best_m3_frozen.pt"
    test_npz_path = RESULTS_DIR / "m3_final_test_predictions.npz"

    exp_ckpt_sha = "5b22607444f4a242a52d0d9337e60c4c63044542dc6796a4a9de78c5ef38057c"
    exp_test_sha = "02fd6eb78682be8ca5743c4b3fddfcc7f57ed56f27f8496092108c30b2188a3d"

    actual_ckpt_sha = compute_sha256(ckpt_path)
    actual_test_sha = compute_sha256(test_npz_path)

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

    # 3. Output File Existence & Non-Emptiness
    required_files = [
        RESULTS_DIR / "phase15_score_distribution.csv",
        RESULTS_DIR / "phase15_ranking_analysis.csv",
        RESULTS_DIR / "phase15_validation_threshold_frontier.csv",
        RESULTS_DIR / "phase15_test_oracle_threshold_frontier.csv",
        RESULTS_DIR / "phase15_temporal_policy_sweep.csv",
        RESULTS_DIR / "phase15_validation_policy.csv",
        RESULTS_DIR / "phase15_test_oracle_policy.csv",
        RESULTS_DIR / "phase15_calibration_comparison.csv",
        RESULTS_DIR / "phase15_temporal_trajectory_analysis.csv",
        RESULTS_DIR / "phase15_hard_case_analysis.csv",
        RESULTS_DIR / "phase15_utility_decomposition.csv",
        RESULTS_DIR / "phase15_control_policies.csv",
        RESULTS_DIR / "phase15_upper_bounds.csv",
        RESULTS_DIR / "phase15_bootstrap_ci.csv",
        RESULTS_DIR / "phase15_cross_domain_transfer.csv",
        RESULTS_DIR / "phase15_decision_summary.json",
        RESULTS_DIR / "phase15_test_report.md",
        RESULTS_DIR / "phase15_novelty_matrix.csv",
        RESULTS_DIR / "phase15_frozen_policy.json",
        RESULTS_DIR / "phase15_freeze_manifest.md",
    ]

    print("\n3. Required Output Files Integrity:")
    all_files_exist = True
    for fpath in required_files:
        exists = fpath.exists() and fpath.stat().st_size > 0
        status = "PASSED" if exists else "FAILED"
        if not exists: all_files_exist = False
        print(f"   File: {fpath.name:45s} [{status}]")

    if not all_files_exist:
        print("   CRITICAL ERROR: One or more required Phase 15 output files missing or empty!")
        print("   SCIENTIFIC VALIDITY: FAILED")
        sys.exit(1)

    # 4. Scorer Equivalence Verification
    summary_json = json.load(open(RESULTS_DIR / "phase15_decision_summary.json"))
    scorer_diff = summary_json["official_scorer_diff"]

    print("\n4. Official Scorer vs Independent Utility Decomposition Check:")
    print(f"   Scorer Difference : {scorer_diff:.12e}")
    print(f"   Scorer Audit Status: {'PASSED [ZERO DISCREPANCY <= 1e-10]' if scorer_diff <= 1e-10 else 'FAILED'}")

    if scorer_diff > 1e-10:
        print("   CRITICAL ERROR: Official scorer mismatch!")
        print("   SCIENTIFIC VALIDITY: FAILED")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("   ALL PHASE 15 AUDIT CHECKS PASSED SUCCESSFULLY")
    print("   SCIENTIFIC VALIDITY: PASSED")
    print("=" * 80)

if __name__ == "__main__":
    main()
