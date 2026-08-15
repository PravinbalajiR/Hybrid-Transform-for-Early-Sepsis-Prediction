"""
audit_raw_validation_sweep.py
-----------------------------
Audit of Raw Validation Prediction Threshold Sweep.
Reads m3_validation_threshold_sweep.csv and m3_selected_thresholds.json to verify
exact validation threshold selection logic, U_val(0.44) vs U_val(0.60), argmax U_val,
and tie-breaking rules.
"""

import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"

def main():
    print("=" * 80)
    print("   RAW VALIDATION THRESHOLD SELECTION AUDIT")
    print("=" * 80)

    val_sweep_path = RESULTS_DIR / "m3_validation_threshold_sweep.csv"
    thresh_json_path = RESULTS_DIR / "m3_selected_thresholds.json"

    print(f"\n1. Validation Sweep Path : {val_sweep_path}")
    print(f"   Exists               : {val_sweep_path.exists()}")

    if not val_sweep_path.exists():
        print("   Error: m3_validation_threshold_sweep.csv not found!")
        return

    df_val = pd.read_csv(val_sweep_path)
    print(f"\n2. Loaded Validation Sweep Records: {len(df_val)} thresholds (from 0.01 to 0.99)")

    # Inspect U_val at specific operating points
    row_44 = df_val.iloc[(df_val['threshold'] - 0.44).abs().argsort()[:1]].iloc[0]
    row_60 = df_val.iloc[(df_val['threshold'] - 0.60).abs().argsort()[:1]].iloc[0]
    row_78 = df_val.iloc[(df_val['threshold'] - 0.78).abs().argsort()[:1]].iloc[0]

    print(f"\n3. Validation Utility Comparison:")
    print(f"   At th = 0.44 --> Validation Utility: {row_44['utility']:+.4f} | F1: {row_44['f1']:.4f} | Prec: {row_44['precision']:.4f} | Rec: {row_44['recall']:.4f} | FPR: {row_44['fpr']:.4f}")
    print(f"   At th = 0.60 --> Validation Utility: {row_60['utility']:+.4f} | F1: {row_60['f1']:.4f} | Prec: {row_60['precision']:.4f} | Rec: {row_60['recall']:.4f} | FPR: {row_60['fpr']:.4f}")
    print(f"   At th = 0.78 --> Validation Utility: {row_78['utility']:+.4f} | F1: {row_78['f1']:.4f} | Prec: {row_78['precision']:.4f} | Rec: {row_78['recall']:.4f} | FPR: {row_78['fpr']:.4f}")

    # Find argmax Utility on Validation Set
    idx_max_u = df_val['utility'].idxmax()
    row_max_u = df_val.loc[idx_max_u]
    print(f"\n4. Validation argmax(Utility):")
    print(f"   Optimal Threshold : th = {row_max_u['threshold']:.4f}")
    print(f"   Validation Utility: {row_max_u['utility']:+.4f}")
    print(f"   Validation F1     : {row_max_u['f1']:.4f}")
    print(f"   Validation Prec   : {row_max_u['precision']:.4f}")
    print(f"   Validation Rec    : {row_max_u['recall']:.4f}")

    # Find argmax F1 on Validation Set
    idx_max_f1 = df_val['f1'].idxmax()
    row_max_f1 = df_val.loc[idx_max_f1]
    print(f"\n5. Validation argmax(F1):")
    print(f"   Optimal Threshold : th = {row_max_f1['threshold']:.4f}")
    print(f"   Validation Utility: {row_max_f1['utility']:+.4f}")
    print(f"   Validation F1     : {row_max_f1['f1']:.4f}")

    if thresh_json_path.exists():
        selected_json = json.loads(thresh_json_path.read_text())
        print(f"\n6. Threshold JSON Artifact (m3_selected_thresholds.json):")
        print(f"   {selected_json}")

    print("\n" + "=" * 80)
    print("   VALIDATION AUDIT COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
