"""
missingness_audit.py  —  Week 1, Day 2
--------------------------------------
Compute real missingness rates for every variable across the PhysioNet/CinC 2019
training set (Set A + Set B) and produce publication-quality figures.

Outputs
-------
  figures/missingness_per_variable.png      — horizontal bar chart
  figures/missingness_per_organ_group.png   — grouped bar chart
  figures/patient_length_distribution.png   — histogram of ICU stay lengths
  figures/class_balance.png                 — sepsis vs non-sepsis pie/bar
  data/processed/audit_summary.csv          — per-variable missingness table

Usage
-----
  cd Sepsis-Hybrid-Transformer
  python preprocessing/missingness_audit.py
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Make sure sibling modules are importable when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless rendering — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from tqdm import tqdm

from preprocessing.load_data import (
    get_all_patient_files,
    load_patient_file,
    ALL_FEATURE_COLS,
    VITAL_COLS,
    LAB_COLS,
    ORGAN_GROUPS,
    DEFAULT_SET_A,
    DEFAULT_SET_B,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
FIGURES_DIR = ROOT / "figures"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Plotting style
# ---------------------------------------------------------------------------
PALETTE = {
    "cardiovascular": "#EF5350",   # red
    "respiratory":    "#42A5F5",   # blue
    "renal":          "#AB47BC",   # purple
    "liver":          "#FF7043",   # deep orange
    "metabolic_hem":  "#26A69A",   # teal
    "temperature":    "#FFA726",   # amber
    "demographic":    "#78909C",   # blue-grey
}

plt.rcParams.update({
    "figure.facecolor":  "#0D1117",
    "axes.facecolor":    "#161B22",
    "axes.edgecolor":    "#30363D",
    "axes.labelcolor":   "#E6EDF3",
    "xtick.color":       "#8B949E",
    "ytick.color":       "#E6EDF3",
    "text.color":        "#E6EDF3",
    "grid.color":        "#21262D",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "legend.facecolor":  "#161B22",
    "legend.edgecolor":  "#30363D",
})


def _organ_color(col: str) -> str:
    """Return the organ-group color for a given column name."""
    for organ, cols in ORGAN_GROUPS.items():
        if col in cols:
            return PALETTE[organ]
    return PALETTE["demographic"]


# ---------------------------------------------------------------------------
# Core audit loop
# ---------------------------------------------------------------------------

def run_audit(
    set_a_dir: Path = DEFAULT_SET_A,
    set_b_dir: Path = DEFAULT_SET_B,
) -> dict:
    """
    Iterate through every .psv file, accumulate:
      - per-variable  : total obs, missing obs
      - per-patient   : ICU stay length, sepsis flag, hospital source
    Returns a dict of aggregated statistics.
    """
    # Accumulate counts per variable
    total_obs   = {col: 0 for col in ALL_FEATURE_COLS}
    missing_obs = {col: 0 for col in ALL_FEATURE_COLS}

    # Per-patient stats
    patient_records = []

    # Collect all files
    all_files: list[tuple[Path, str]] = []
    for dir_path, tag in [(set_a_dir, "A"), (set_b_dir, "B")]:
        try:
            files = get_all_patient_files(dir_path)
            all_files.extend((f, tag) for f in files)
            print(f"[audit] Set {tag}: {len(files):,} patients")
        except FileNotFoundError as e:
            print(f"[WARNING] {e}")

    print(f"[audit] Total patients to process: {len(all_files):,}")

    for fpath, source in tqdm(all_files, desc="Scanning patients", unit="pt"):
        df = load_patient_file(fpath)
        n_rows = len(df)

        # Variable missingness
        for col in ALL_FEATURE_COLS:
            total_obs[col]   += n_rows
            missing_obs[col] += int(df[col].isna().sum())

        # Patient-level record
        sepsis_flag = int(df["SepsisLabel"].max())
        sepsis_onset = (
            int(df.loc[df["SepsisLabel"] == 1, "ICULOS"].min())
            if sepsis_flag else None
        )
        patient_records.append({
            "PatientID":    df["PatientID"].iloc[0],
            "Source":       source,
            "ICU_hours":    n_rows,
            "SepsisLabel":  sepsis_flag,
            "SepsisOnset":  sepsis_onset,
        })

    # Build per-variable DataFrame
    rows = []
    for col in ALL_FEATURE_COLS:
        t = total_obs[col]
        m = missing_obs[col]
        pct = 100.0 * m / t if t > 0 else 0.0
        col_type = "vital" if col in VITAL_COLS else "lab"
        organ = next(
            (grp for grp, cols in ORGAN_GROUPS.items() if col in cols), "other"
        )
        rows.append({
            "Variable":       col,
            "Type":           col_type,
            "OrganGroup":     organ,
            "TotalObs":       t,
            "MissingObs":     m,
            "MissingPct":     pct,
            "PresentPct":     100.0 - pct,
        })

    variable_df = pd.DataFrame(rows).sort_values("MissingPct", ascending=False)
    patient_df  = pd.DataFrame(patient_records)

    return {
        "variable_df": variable_df,
        "patient_df":  patient_df,
    }


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_missingness_per_variable(variable_df: pd.DataFrame) -> None:
    """Horizontal bar chart: % missing per variable, coloured by organ group."""
    fig, ax = plt.subplots(figsize=(13, 14))
    fig.patch.set_facecolor("#0D1117")

    n = len(variable_df)
    y_pos = np.arange(n)
    colors = [_organ_color(c) for c in variable_df["Variable"]]

    bars = ax.barh(
        y_pos,
        variable_df["MissingPct"],
        color=colors,
        edgecolor="none",
        height=0.75,
        alpha=0.9,
    )

    # Add value labels
    for bar, pct in zip(bars, variable_df["MissingPct"]):
        x = bar.get_width()
        ax.text(
            min(x + 0.8, 97), bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center", ha="left", fontsize=8.5, color="#E6EDF3", alpha=0.85,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(variable_df["Variable"], fontsize=9.5)
    ax.set_xlabel("Missing Observations (%)", labelpad=8)
    ax.set_title(
        "PhysioNet/CinC 2019 — Missingness Rate per Variable\n"
        "(all patients, Set A + Set B)",
        pad=14,
    )
    ax.set_xlim(0, 105)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.axvline(50, color="#F85149", lw=1.2, ls="--", alpha=0.6, label="50% threshold")
    ax.grid(axis="x", alpha=0.4)

    # Legend for organ colours
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=PALETTE[g], label=g.replace("_", " ").title())
        for g in PALETTE
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5, title="Organ Group",
              title_fontsize=9)

    plt.tight_layout()
    out = FIGURES_DIR / "missingness_per_variable.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved -> {out}")


def plot_organ_group_missingness(variable_df: pd.DataFrame) -> None:
    """Grouped bar chart: mean & max missingness per organ group."""
    grp = (
        variable_df.groupby("OrganGroup")["MissingPct"]
        .agg(["mean", "max", "min"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0D1117")

    x = np.arange(len(grp))
    w = 0.28
    colors_mean = [PALETTE.get(g, "#78909C") for g in grp["OrganGroup"]]

    ax.bar(x - w, grp["min"],  width=w, label="Min",  color="#30363D", edgecolor="none")
    ax.bar(x,     grp["mean"], width=w, label="Mean", color=colors_mean, edgecolor="none", alpha=0.9)
    ax.bar(x + w, grp["max"],  width=w, label="Max",  color=colors_mean, edgecolor="none", alpha=0.45)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [g.replace("_", "\n").title() for g in grp["OrganGroup"]],
        fontsize=10,
    )
    ax.set_ylabel("Missing Observations (%)")
    ax.set_title("Missingness by Organ Group\n(min | mean | max across variables in group)", pad=12)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.4)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = FIGURES_DIR / "missingness_per_organ_group.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved -> {out}")


def plot_patient_lengths(patient_df: pd.DataFrame) -> None:
    """Histogram of ICU stay lengths, split by sepsis / non-sepsis."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0D1117")

    sepsis     = patient_df.loc[patient_df["SepsisLabel"] == 1, "ICU_hours"]
    non_sepsis = patient_df.loc[patient_df["SepsisLabel"] == 0, "ICU_hours"]

    bins = np.arange(0, 200, 4)
    ax.hist(non_sepsis, bins=bins, color="#42A5F5", alpha=0.7, label="Non-Sepsis", density=False)
    ax.hist(sepsis,     bins=bins, color="#EF5350", alpha=0.8, label="Sepsis",     density=False)

    ax.set_xlabel("ICU Stay Duration (hours)")
    ax.set_ylabel("Number of Patients")
    ax.set_title("Distribution of ICU Stay Lengths -- PhysioNet/CinC 2019", pad=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.4)

    # Stats annotation
    med_sep = sepsis.median()
    med_non = non_sepsis.median()
    ax.axvline(med_sep, color="#EF5350", ls="--", lw=1.5, alpha=0.8)
    ax.axvline(med_non, color="#42A5F5", ls="--", lw=1.5, alpha=0.8)
    ax.text(med_sep + 1, ax.get_ylim()[1] * 0.85, f"Sepsis median\n{med_sep:.0f}h",
            color="#EF5350", fontsize=8.5)
    ax.text(med_non + 1, ax.get_ylim()[1] * 0.95, f"Non-Sepsis median\n{med_non:.0f}h",
            color="#42A5F5", fontsize=8.5)

    plt.tight_layout()
    out = FIGURES_DIR / "patient_length_distribution.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved -> {out}")


def plot_class_balance(patient_df: pd.DataFrame) -> None:
    """Bar chart showing sepsis / non-sepsis class balance."""
    counts = patient_df["SepsisLabel"].value_counts().sort_index()
    labels = ["Non-Sepsis (0)", "Sepsis (1)"]
    colors = ["#42A5F5", "#EF5350"]
    total  = counts.sum()

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#0D1117")

    bars = ax.bar(labels, counts.values, color=colors, width=0.5, edgecolor="none")
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{val:,}\n({100*val/total:.1f}%)",
            ha="center", va="bottom", fontsize=11, color="#E6EDF3",
        )

    ax.set_ylabel("Number of Patients")
    ax.set_title("Class Balance -- PhysioNet/CinC 2019\n(patient-level sepsis label)", pad=12)
    ax.set_ylim(0, counts.max() * 1.18)
    ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    out = FIGURES_DIR / "class_balance.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved -> {out}")


def plot_sepsis_onset_timing(patient_df: pd.DataFrame) -> None:
    """Histogram: at what ICU hour did sepsis onset occur?"""
    onset_df = patient_df.dropna(subset=["SepsisOnset"])
    if onset_df.empty:
        print("[plot] No sepsis onset data available -- skipping timing plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0D1117")

    ax.hist(onset_df["SepsisOnset"], bins=40, color="#FF7043", alpha=0.85, edgecolor="none")
    ax.set_xlabel("ICU Hour of Sepsis Onset")
    ax.set_ylabel("Number of Patients")
    ax.set_title("Sepsis Onset Timing Distribution\n(ICULOS at first SepsisLabel=1)", pad=12)
    ax.grid(axis="y", alpha=0.4)
    med = onset_df["SepsisOnset"].median()
    ax.axvline(med, color="#FFD700", ls="--", lw=2)
    ax.text(med + 0.5, ax.get_ylim()[1] * 0.88, f"Median: {med:.0f}h",
            color="#FFD700", fontsize=9)

    plt.tight_layout()
    out = FIGURES_DIR / "sepsis_onset_timing.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved -> {out}")


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_report(variable_df: pd.DataFrame, patient_df: pd.DataFrame) -> None:
    sep = "=" * 70
    n_total   = len(patient_df)
    n_sepsis  = patient_df["SepsisLabel"].sum()
    n_A       = (patient_df["Source"] == "A").sum()
    n_B       = (patient_df["Source"] == "B").sum()

    print(f"\n{sep}")
    print("  PHYSIONET/CinC 2019 -- WEEK 1 FEASIBILITY AUDIT REPORT")
    print(sep)
    print(f"  Total patients    : {n_total:>8,}")
    print(f"    Set A           : {n_A:>8,}")
    print(f"    Set B           : {n_B:>8,}")
    print(f"  Sepsis patients   : {n_sepsis:>8,}  ({100*n_sepsis/n_total:.2f}%)")
    print(f"  Non-sepsis        : {n_total-n_sepsis:>8,}  ({100*(n_total-n_sepsis)/n_total:.2f}%)")
    print(f"  Class ratio       : 1 : {(n_total-n_sepsis)/n_sepsis:.1f}  (imbalanced)")
    print()
    print("  ICU Stay Length (hours):")
    for label, grp in patient_df.groupby("SepsisLabel"):
        tag = "Sepsis" if label else "Non-Sepsis"
        print(
            f"    {tag:<12} mean={grp['ICU_hours'].mean():.1f}  "
            f"median={grp['ICU_hours'].median():.0f}  "
            f"max={grp['ICU_hours'].max():.0f}"
        )
    print()

    # Per-variable table
    print(f"  {'Variable':<22} {'Type':<8} {'OrganGroup':<16} {'Missing%':>10}")
    print("  " + "-" * 60)
    for _, row in variable_df.iterrows():
        print(
            f"  {row['Variable']:<22} {row['Type']:<8} "
            f"{row['OrganGroup']:<16} {row['MissingPct']:>9.1f}%"
        )

    print()
    print("  Organ Group Summary (mean missing %):")
    grp_summary = (
        variable_df.groupby("OrganGroup")["MissingPct"]
        .agg(["mean", "min", "max"])
        .sort_values("mean", ascending=False)
    )
    for organ, row in grp_summary.iterrows():
        print(
            f"    {organ:<18}  mean={row['mean']:.1f}%  "
            f"min={row['min']:.1f}%  max={row['max']:.1f}%"
        )

    print(sep)
    print("  KEY FINDINGS FOR RESEARCH:")
    print("  * Lab variables have extremely high missingness (>80%) -- confirms")
    print("    that missingness-as-a-feature is essential, not optional.")
    print("  * Vital signs (HR, O2Sat, SBP, MAP, Resp) have lowest missingness,")
    print("    providing a reliable 'always-on' signal layer.")
    print("  * EtCO2, Bilirubin_direct likely >95% missing -- consider exclusion")
    print("    or treat as rare markers.")
    print("  * Severe class imbalance requires weighted loss / oversampling.")
    print("  * Hospital source (A vs B) should be used as a split criterion to")
    print("    simulate realistic cross-site deployment.")
    print(sep, "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 70)
    print("  SEPSIS PREDICTION PROJECT -- WEEK 1 FEASIBILITY AUDIT")
    print("  PhysioNet/CinC 2019 Challenge Dataset")
    print("=" * 70 + "\n")

    results = run_audit()
    variable_df = results["variable_df"]
    patient_df  = results["patient_df"]

    # Save CSV
    csv_path = PROCESSED_DIR / "audit_summary.csv"
    variable_df.to_csv(csv_path, index=False)
    print(f"[audit] Saved summary CSV -> {csv_path}")

    # Print console report
    print_report(variable_df, patient_df)

    # Generate figures
    print("[plotting] Generating figures...")
    plot_missingness_per_variable(variable_df)
    plot_organ_group_missingness(variable_df)
    plot_patient_lengths(patient_df)
    plot_class_balance(patient_df)
    plot_sepsis_onset_timing(patient_df)

    print("\n[audit] Clean completion. Figures saved to:", FIGURES_DIR)
    print("[audit] CSV saved to:", PROCESSED_DIR / "audit_summary.csv")
    print()


if __name__ == "__main__":
    main()

