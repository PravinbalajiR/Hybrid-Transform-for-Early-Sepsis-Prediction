import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from pathlib import Path

# Set publication style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 13

base_dir = Path('.').resolve()
fig_dir = base_dir / 'figures'
fig_dir.mkdir(parents=True, exist_ok=True)
submission_fig_dir = base_dir / 'submission_package' / 'figures'
submission_fig_dir.mkdir(parents=True, exist_ok=True)

print("=== GENERATING PUBLICATION FIGURES ===")

# ---------------------------------------------------------
# FIGURE 1: DISCRIMINATION (ROC & PR CURVES)
# ---------------------------------------------------------
npz_path = base_dir / 'results/m3_final_test_predictions.npz'
data = np.load(npz_path, allow_pickle=True)
y_true = data['y_true_flat']
y_proba = data['y_proba_flat']

fpr, tpr, _ = roc_curve(y_true, y_proba)
auroc_val = auc(fpr, tpr)

precision, recall, _ = precision_recall_curve(y_true, y_proba)
auprc_val = auc(recall, precision)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)

# Panel A: ROC Curve
ax1.plot(fpr, tpr, color='#1f77b4', lw=2.2, label=f'M3 Time-Aware Transformer (AUROC = {auroc_val:.4f})')
ax1.plot([0, 1], [0, 1], color='#7f7f7f', lw=1.2, linestyle='--', label='Random Chance Baseline (AUROC = 0.5000)')
ax1.set_xlim([-0.02, 1.02])
ax1.set_ylim([-0.02, 1.02])
ax1.set_xlabel('False Positive Rate (1 - Specificity)')
ax1.set_ylabel('True Positive Rate (Sensitivity)')
ax1.set_title('A: Receiver Operating Characteristic (ROC)', fontweight='bold', loc='left')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)

# Panel B: Precision-Recall Curve
baseline_pr = np.mean(y_true)
ax2.plot(recall, precision, color='#2ca02c', lw=2.2, label=f'M3 Time-Aware Transformer (AUPRC = {auprc_val:.4f})')
ax2.axhline(baseline_pr, color='#7f7f7f', lw=1.2, linestyle='--', label=f'Random Prevalence Baseline ({baseline_pr:.4f})')
ax2.set_xlim([-0.02, 1.02])
ax2.set_ylim([-0.02, 1.02])
ax2.set_xlabel('Recall (Sensitivity)')
ax2.set_ylabel('Precision (Positive Predictive Value)')
ax2.set_title('B: Precision-Recall (PR) Curve', fontweight='bold', loc='left')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

plt.tight_layout()

for ext in ['png', 'pdf', 'svg']:
    p1 = fig_dir / f'figure1_discrimination.{ext}'
    p2 = submission_fig_dir / f'figure1_discrimination.{ext}'
    fig.savefig(p1, dpi=300, bbox_inches='tight')
    fig.savefig(p2, dpi=300, bbox_inches='tight')
    print(f'Saved Figure 1 ({ext}): {p1.name}')

plt.close(fig)

# ---------------------------------------------------------
# FIGURE 2: DECISION UTILITY VS THRESHOLD
# ---------------------------------------------------------
thresholds = np.array([0.05, 0.10, 0.15, 0.19, 0.25, 0.30, 0.31, 0.50, 0.70])
utilities = np.array([0.520426, 0.622666, 0.654351, 0.655944, 0.640674, 0.621805, 0.620532, 0.583403, 0.517381])

fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)

ax.plot(thresholds, utilities, marker='o', color='#d62728', lw=2.2, ms=6, label='Descriptive Test-Set Sensitivity Sweep')
ax.axvline(0.190, color='#1f77b4', lw=1.8, linestyle='--', label='Prespecified Validation-Selected Threshold (th = 0.190)')
ax.plot(0.190, 0.655944, marker='*', color='#1f77b4', ms=14, zorder=5)

ax.annotate('Prespecified Operating Point\nth = 0.190 (U_official = +0.655944)',
            xy=(0.190, 0.655944), xytext=(0.28, 0.640),
            arrowprops=dict(facecolor='#1f77b4', shrink=0.08, width=1.5, headwidth=8),
            fontweight='bold', fontsize=9.5, bbox=dict(boxstyle='round,pad=0.4', facecolor='#f7f7f7', edgecolor='#1f77b4'))

ax.set_xlim([0.02, 0.73])
ax.set_ylim([0.48, 0.68])
ax.set_xlabel('Decision Probability Threshold (th)')
ax.set_ylabel('Official PhysioNet 2019 Normalized Utility (U_official)')
ax.set_title('Official PhysioNet 2019 Utility Across Decision Thresholds\n(Emory External Test Cohort, N=20,000 Stays)', fontweight='bold', loc='left')
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)

plt.tight_layout()

for ext in ['png', 'pdf', 'svg']:
    p1 = fig_dir / f'figure2_utility_threshold.{ext}'
    p2 = submission_fig_dir / f'figure2_utility_threshold.{ext}'
    fig.savefig(p1, dpi=300, bbox_inches='tight')
    fig.savefig(p2, dpi=300, bbox_inches='tight')
    print(f'Saved Figure 2 ({ext}): {p2.name}')

plt.close(fig)

# ---------------------------------------------------------
# FIGURE 3: OPERATIONAL ALERT BURDEN
# ---------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4.2), dpi=300)

# Panel A: Alert Composition
labels_a = ['True Positive\n(1,004 Alerts\n18.81%)', 'False Positive\n(4,333 Alerts\n81.19%)']
sizes_a = [1004, 4333]
colors_a = ['#2ca02c', '#d62728']
ax1.pie(sizes_a, labels=labels_a, colors=colors_a, autopct='', startangle=140, explode=(0.05, 0),
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2))
ax1.set_title('A: Total Alerts Composition\n(5,337 Total Alerts)', fontweight='bold')

# Panel B: Alert Frequency
categories_b = ['Alert Rate\n(per 100 Patient-Days)']
values_b = [16.99]
bars_b = ax2.bar(categories_b, values_b, color='#1f77b4', width=0.4, edgecolor='black', linewidth=1)
ax2.set_ylabel('Alerts per 100 Patient-Days')
ax2.set_ylim([0, 22])
ax2.set_title('B: Operational Alert Frequency', fontweight='bold')
ax2.grid(axis='y', linestyle=':', alpha=0.6)
for bar in bars_b:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.6, f'{yval:.2f}', ha='center', va='bottom', fontweight='bold')

# Panel C: Patient Coverage
labels_c = ['Alerted Stays\n(5,172 Stays\n25.86%)', 'Non-Alerted Stays\n(14,828 Stays\n74.14%)']
sizes_c = [5172, 14828]
colors_c = ['#ff7f0e', '#bcbd22']
ax3.pie(sizes_c, labels=labels_c, colors=colors_c, autopct='', startangle=140, explode=(0.05, 0),
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2))
ax3.set_title('C: Patient Alert Coverage\n(N=20,000 Emory Stays)', fontweight='bold')

plt.tight_layout()

for ext in ['png', 'pdf', 'svg']:
    p1 = fig_dir / f'figure3_alert_burden.{ext}'
    p2 = submission_fig_dir / f'figure3_alert_burden.{ext}'
    fig.savefig(p1, dpi=300, bbox_inches='tight')
    fig.savefig(p2, dpi=300, bbox_inches='tight')
    print(f'Saved Figure 3 ({ext}): {p3 if "p3" in locals() else p2.name}')

plt.close(fig)

print("=== ALL PUBLICATION FIGURES SUCCESSFULLY GENERATED ===")
