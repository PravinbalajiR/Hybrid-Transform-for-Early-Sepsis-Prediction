"""
organ_groups.py
---------------
The "knowledge layer" of the hybrid model.

This file encodes domain expertise about which ICU variables map to which
organ system. It is the ONLY place where clinical knowledge is hard-coded —
the rest of the model is purely data-driven.

Variables are assigned based on standard critical care physiology:
  - Cardiovascular : perfusion pressure, heart rate, troponin
  - Respiratory    : gas exchange, oxygenation, ventilation
  - Renal          : filtration, electrolyte balance
  - Liver          : synthetic function, cholestasis
  - Metabolic/Hem  : metabolism, coagulation, haematology
  - Temperature    : thermoregulation (isolated group)

NOTE: Glasgow Coma Scale and vasopressor dosing ARE ABSENT from PhysioNet
2019. Full SOFA score cannot be computed. This module reflects the actual
available variables.
"""

from __future__ import annotations
from typing import Dict, List

# ---------------------------------------------------------------------------
# Primary organ-variable mapping
# ---------------------------------------------------------------------------
ORGAN_GROUPS: Dict[str, List[str]] = {
    "cardiovascular": [
        "HR",          # Heart Rate (bpm)
        "SBP",         # Systolic Blood Pressure (mmHg)
        "MAP",         # Mean Arterial Pressure (mmHg)
        "DBP",         # Diastolic Blood Pressure (mmHg)
        "TroponinI",   # Cardiac troponin I (ng/mL)
    ],
    "respiratory": [
        "O2Sat",       # Pulse oximetry (%)
        "Resp",        # Respiratory Rate (breaths/min)
        "EtCO2",       # End-tidal CO2 (mmHg)
        "FiO2",        # Fraction of inspired O2 (0–1)
        "PaCO2",       # Arterial CO2 tension (mmHg)
        "SaO2",        # Arterial O2 saturation (%)
        "pH",          # Arterial blood pH
    ],
    "renal": [
        "Creatinine",  # Serum creatinine (mg/dL)
        "BUN",         # Blood urea nitrogen (mg/dL)
        "Chloride",    # Serum chloride (mEq/L)
        "Calcium",     # Serum calcium (mg/dL)
        "Potassium",   # Serum potassium (mEq/L)
        "Magnesium",   # Serum magnesium (mEq/L)
        "Phosphate",   # Serum phosphate (mg/dL)
    ],
    "liver": [
        "AST",               # Aspartate aminotransferase (IU/L)
        "Alkalinephos",      # Alkaline phosphatase (IU/L)
        "Bilirubin_direct",  # Direct bilirubin (mg/dL)
        "Bilirubin_total",   # Total bilirubin (mg/dL)
    ],
    "metabolic_hem": [
        "Glucose",     # Blood glucose (mg/dL)
        "Lactate",     # Serum lactate (mmol/L)   ← key sepsis marker
        "BaseExcess",  # Base excess (mEq/L)
        "HCO3",        # Serum bicarbonate (mEq/L)
        "WBC",         # White blood cell count (10^3/µL)
        "Hct",         # Haematocrit (%)
        "Hgb",         # Haemoglobin (g/dL)
        "PTT",         # Partial thromboplastin time (seconds)
        "Fibrinogen",  # Fibrinogen (mg/dL)
        "Platelets",   # Platelet count (10^3/µL)
    ],
    "temperature": [
        "Temp",        # Temperature (°C)  ← SIRS criterion
    ],
}

# Flat list preserving organ-group order (used as the canonical feature order)
ORGAN_ORDERED_COLS: List[str] = [
    col for cols in ORGAN_GROUPS.values() for col in cols
]

# Reverse map: variable name → organ group
VAR_TO_ORGAN: Dict[str, str] = {
    col: organ
    for organ, cols in ORGAN_GROUPS.items()
    for col in cols
}

# Group sizes (F_k for each organ k)
ORGAN_GROUP_SIZES: Dict[str, int] = {
    organ: len(cols) for organ, cols in ORGAN_GROUPS.items()
}

# ---------------------------------------------------------------------------
# Derived features (computed in preprocessing, not in raw PSV)
# ---------------------------------------------------------------------------
DERIVED_FEATURES: Dict[str, str] = {
    "ShockIndex": "HR / SBP — ratio > 1.0 predicts haemodynamic instability",
    # "PaO2_FiO2": "PaO2/FiO2 ratio — requires PaO2 which is absent; use SaO2/FiO2",
    # "SaO2_FiO2": "SaO2 / FiO2 — available surrogate for P/F ratio",
}

# ---------------------------------------------------------------------------
# qSOFA criteria (partial — GCS absent)
# ---------------------------------------------------------------------------
# qSOFA = Respiratory Rate ≥ 22 + SBP ≤ 100 + GCS < 15
# GCS is NOT in PhysioNet 2019 → only 2/3 qSOFA criteria computable
PARTIAL_QSOFA_AVAILABLE = ["Resp", "SBP"]
PARTIAL_QSOFA_MISSING   = ["GCS"]   # confirmed absent from dataset

# ---------------------------------------------------------------------------
# SIRS criteria (partial — temperature thresholds apply)
# ---------------------------------------------------------------------------
SIRS_CRITERIA = {
    "Temp":  "< 36°C or > 38°C",
    "HR":    "> 90 bpm",
    "Resp":  "> 20 breaths/min",
    "WBC":   "< 4 or > 12 × 10^3/µL",
}


def get_organ_variable_indices(feature_cols: List[str]) -> Dict[str, List[int]]:
    """
    Given a feature column list (in whatever order the model uses),
    return the integer indices that correspond to each organ group.

    Parameters
    ----------
    feature_cols : ordered list of feature names (as used in the model input)

    Returns
    -------
    Dict mapping organ name → list of indices into feature_cols
    """
    col_to_idx = {col: i for i, col in enumerate(feature_cols)}
    indices: Dict[str, List[int]] = {}
    for organ, cols in ORGAN_GROUPS.items():
        idxs = [col_to_idx[c] for c in cols if c in col_to_idx]
        if idxs:
            indices[organ] = idxs
    return indices


def print_organ_summary() -> None:
    """Pretty-print the organ grouping for inspection."""
    header = "=" * 60
    print(f"\n{header}")
    print("  ORGAN VARIABLE GROUPS  (PhysioNet/CinC 2019)")
    print(header)
    for organ, cols in ORGAN_GROUPS.items():
        print(f"\n  [{organ.upper().replace('_',' ')}]  ({len(cols)} variables)")
        for c in cols:
            print(f"    • {c}")
    print(f"\n  TOTAL: {len(ORGAN_ORDERED_COLS)} variables across "
          f"{len(ORGAN_GROUPS)} organ groups")
    print(f"\n  UNAVAILABLE (dataset limitation):")
    print(f"    • Glasgow Coma Scale (GCS) — absent from PhysioNet 2019")
    print(f"    • Vasopressor dosing        — absent from PhysioNet 2019")
    print(f"    → Full clinical SOFA cannot be computed")
    print(header + "\n")


if __name__ == "__main__":
    print_organ_summary()

    from preprocessing.load_data import ALL_FEATURE_COLS
    indices = get_organ_variable_indices(ALL_FEATURE_COLS)
    print("Variable indices per organ group:")
    for organ, idxs in indices.items():
        print(f"  {organ:<18}: indices {idxs}")
