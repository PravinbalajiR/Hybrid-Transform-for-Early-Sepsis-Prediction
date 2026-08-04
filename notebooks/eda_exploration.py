"""
eda_exploration.py  —  Week 2-3 Exploratory Data Analysis
----------------------------------------------------------
Run after the Week 1 missingness audit. Uses the audit_summary.csv results
to produce additional EDA plots:

  1. Variable correlation heatmap (on observed values)
  2. Missingness pattern co-occurrence matrix
  3. Time-series sample plots (sepsis vs non-sepsis patients)
  4. Lab draw frequency histograms (when do clinicians draw labs?)

Usage
-----
  cd Sepsis-Hybrid-Transformer
  python notebooks/eda_exploration.py
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm

from preprocessing.load_data import (
    load_dataset, ALL_FEATURE_COLS, VITAL_COLS, LAB_COLS,
    ORGAN_GROUPS, DEFAULT_SET_A, DEFAULT_SET_B,
)

ROOT        = Path(__file__).parent.parent
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#0D1117",
    "axes.facecolor":   "#161B22",
    "axes.edgecolor":   "#30363D",
    "axes.labelcolor":  "#E6EDF3",
    "xtick.color":      "#8B949E",
    "ytick.color":      "#8B949E",
    "text.color":       "#E6EDF3",
    "grid.color":       "#21262D",
    "font.family":      "sans-serif",
    "font.size":        10,
})

ORGAN_COLORS = {
    "cardiovascular": "#EF5350",
    "respiratory":    "#42A5F5",
    "renal":          "#AB47BC",
    "liver":          "#FF7043",
    "metabolic_hem":  "#26A69A",
    "temperature":    "#FFA726",
}


def plot_correlation_heatmap(patient_dfs: list, n_patients: int = 2000) -> None:
    """Pearson correlation heatmap on observed vital signs."""
    print("[EDA] Building correlation matrix (vitals)...")
    rows = []
    for df in patient_dfs[:n_patients]:
        rows.append(df[VITAL_COLS])
    combined = pd.concat(rows, ignore_index=True)

    corr = combined.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor("#0D1117")

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        linewidths=0.5,
        linecolor="#21262D",
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(
        f"Vital Sign Pearson Correlation\n(first {n_patients} patients, observed values only)",
        pad=12,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "vital_correlation_heatmap.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[EDA] Saved -> {out}")


def plot_sample_timeseries(patient_dfs: list) -> None:
    """
    Plot 3 vital signs over time for 2 sepsis and 2 non-sepsis patients.
    Shows the challenge: irregular observation, high noise, gradual deterioration.
    """
    sepsis_dfs     = [df for df in patient_dfs if df["SepsisLabel"].max() == 1][:2]
    nonsepsis_dfs  = [df for df in patient_dfs if df["SepsisLabel"].max() == 0][:2]
    sample_dfs     = sepsis_dfs + nonsepsis_dfs

    vitals_to_plot = ["HR", "SBP", "Resp"]
    colors         = ["#EF5350", "#42A5F5", "#26A69A"]

    fig = plt.figure(figsize=(16, 9), facecolor="#0D1117")
    gs  = gridspec.GridSpec(len(vitals_to_plot), 4, figure=fig, hspace=0.5, wspace=0.35)

    col_titles = [
        "Sepsis Patient 1", "Sepsis Patient 2",
        "Non-Sepsis Patient 1", "Non-Sepsis Patient 2"
    ]

    for col_idx, df in enumerate(sample_dfs):
        is_sep = df["SepsisLabel"].max() == 1
        onset  = int(df.loc[df["SepsisLabel"] == 1, "ICULOS"].min()) if is_sep else None

        for row_idx, (vital, color) in enumerate(zip(vitals_to_plot, colors)):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.set_facecolor("#161B22")

            hours = df["ICULOS"].values
            vals  = df[vital].values
            obs   = ~pd.isna(vals)

            ax.plot(hours[obs], vals[obs], color=color, lw=1.5, alpha=0.9)
            ax.scatter(hours[obs], vals[obs], color=color, s=15, alpha=0.6, zorder=5)

            if onset is not None:
                ax.axvline(onset, color="#F85149", lw=1.5, ls="--", alpha=0.8,
                           label="Sepsis onset" if row_idx == 0 else None)

            ax.set_ylabel(vital, fontsize=9)
            if row_idx == 0:
                ax.set_title(
                    col_titles[col_idx],
                    fontsize=10,
                    color="#EF5350" if is_sep else "#42A5F5",
                    fontweight="bold",
                )
            if row_idx == len(vitals_to_plot) - 1:
                ax.set_xlabel("ICU Hour", fontsize=9)

            for spine in ax.spines.values():
                spine.set_edgecolor("#30363D")

    fig.suptitle(
        "Sample ICU Time Series -- Sepsis vs Non-Sepsis Patients\n"
        "(Red dashed line = sepsis onset; gaps = missing observations)",
        fontsize=12, color="#E6EDF3", y=1.01,
    )

    out = FIGURES_DIR / "sample_timeseries.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[EDA] Saved -> {out}")


def plot_lab_draw_frequency(patient_dfs: list, n_patients: int = 5000) -> None:
    """
    For each lab variable, show at which ICU hours observations tend to cluster.
    Reveals rhythmic (morning) lab draws and acute-event-triggered draws.
    """
    print("[EDA] Computing lab draw frequency by hour-of-stay...")

    key_labs = ["Creatinine", "Lactate", "WBC", "Glucose", "Bilirubin_total", "PTT"]
    draw_hours = {lab: [] for lab in key_labs}

    for df in patient_dfs[:n_patients]:
        for lab in key_labs:
            obs_hours = df.loc[df[lab].notna(), "ICULOS"].tolist()
            draw_hours[lab].extend(obs_hours)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor="#0D1117")
    axes_flat = axes.flatten()

    organ_of = {col: grp for grp, cols in ORGAN_GROUPS.items() for col in cols}

    for ax, lab in zip(axes_flat, key_labs):
        ax.set_facecolor("#161B22")
        organ = organ_of.get(lab, "metabolic_hem")
        color = ORGAN_COLORS.get(organ, "#78909C")

        hours = draw_hours[lab]
        if hours:
            bins = np.arange(0, min(max(hours) + 2, 120), 1)
            ax.hist(hours, bins=bins, color=color, alpha=0.85, edgecolor="none")

        ax.set_title(lab, fontsize=11, fontweight="bold")
        ax.set_xlabel("ICU Hour of Stay", fontsize=9)
        ax.set_ylabel("# Observations", fontsize=9)
        ax.set_xlim(0, 120)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363D")

    fig.suptitle(
        f"Lab Draw Frequency by ICU Hour-of-Stay\n(first {n_patients} patients)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "lab_draw_frequency.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[EDA] Saved -> {out}")


def main():
    print("\n[EDA] Loading dataset sample (5000 patients for EDA)...")
    patient_dfs, _ = load_dataset(max_patients=5000, verbose=True)

    plot_correlation_heatmap(patient_dfs, n_patients=2000)
    plot_sample_timeseries(patient_dfs)
    plot_lab_draw_frequency(patient_dfs, n_patients=5000)

    print(f"\n[EDA] All EDA figures saved to {FIGURES_DIR}")



if __name__ == "__main__":
    main()
