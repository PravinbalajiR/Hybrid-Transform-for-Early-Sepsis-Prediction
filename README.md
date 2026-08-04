# A Knowledge-Guided Hybrid Transformer for Early Sepsis Prediction

**Organ-Aware Representations and Missingness Encoding for Irregular ICU Time-Series**

---

## Overview

This project develops a dual-branch hybrid architecture for early sepsis prediction using the
**PhysioNet/CinC 2019 Challenge** dataset (40,336 ICU patients, 34 hourly vitals/labs + demographics,
sepsis-within-6h binary label).

### Key Novelties
1. **Knowledge-guided dual-branch** — organ-physiology branch + time-aware Transformer, fused via cross-attention
2. **Organ-aware representation** — variables grouped into cardiovascular, respiratory, renal, liver, metabolic systems
3. **Missingness-aware temporal encoding** — value + observation mask + time-delta per variable per hour
4. **Uncertainty-aware prediction** — MC Dropout for confidence-gated clinical alerts

---

## Repository Structure

```
Sepsis-Hybrid-Transformer/
│
├── data/
│   ├── raw/                  # Symlinks or copies of training_setA, training_setB
│   ├── processed/            # Preprocessed .npy / .pkl tensors
│   └── splits/               # Train/val/test patient ID lists
│
├── preprocessing/
│   ├── load_data.py          # PSV loader, per-patient DataFrame builder
│   ├── missingness_audit.py  # Week 1: real missingness rates per variable/organ group
│   ├── normalize.py          # Z-score normalization (fit on train only)
│   ├── masks_and_deltas.py   # Observation mask + time-delta computation
│   └── split.py              # Hospital-stratified train/val/test split
│
├── baselines/
│   ├── xgboost_baseline.py   # XGBoost on static aggregated features
│   └── transformer_naive.py  # Plain Transformer with mean imputation
│
├── models/
│   ├── transformer/
│   │   ├── time_aware_embedding.py
│   │   └── transformer_encoder.py
│   ├── organ_branch/
│   │   ├── organ_groups.py   # Variable-to-organ mapping (the knowledge layer)
│   │   └── organ_encoder.py  # Small per-organ MLPs/GRUs
│   └── fusion/
│       ├── cross_attention.py
│       └── hybrid_model.py   # Full dual-branch + fusion + prediction head
│
├── experiments/
│   ├── configs/              # YAML experiment configs
│   ├── logs/                 # Training logs / TensorBoard
│   └── checkpoints/          # Model weights
│
├── evaluation/
│   ├── utility_score.py      # Official PhysioNet 2019 utility metric
│   ├── metrics.py            # AUROC, AUPRC, calibration (ECE)
│   └── case_studies.py       # Qualitative patient-level analysis
│
├── figures/                  # Generated plots for the paper
├── paper/                    # LaTeX / draft text
├── notebooks/                # Exploratory Jupyter notebooks
├── utils/
│   ├── logger.py
│   └── seed.py
│
├── requirements.txt
└── README.md
```

---

## Dataset

- **Source:** [PhysioNet/CinC 2019 Challenge](https://physionet.org/content/challenge-2019/1.0.0/)
- **training_setA** — Hospital A ICU, ~20,336 patients
- **training_setB** — Hospital B ICU, ~20,000 patients
- **Format:** per-patient `.psv` files, hourly rows
- **Confirmed missing from dataset:** Glasgow Coma Scale (GCS), vasopressor dosing → full clinical SOFA **cannot** be computed

### Variable Groups (Organ Mapping)

| Organ System   | Variables |
|----------------|-----------|
| Cardiovascular | HR, SBP, MAP, DBP, TroponinI, Shock Index (HR/SBP) |
| Respiratory    | O2Sat, Resp, EtCO2, FiO2, PaCO2, SaO2, pH, PaO2/FiO2 |
| Renal          | Creatinine, BUN, Chloride, Calcium, Potassium, Magnesium, Phosphate |
| Liver          | AST, Alkalinephos, Bilirubin_direct, Bilirubin_total |
| Metabolic/Hem  | Glucose, Lactate, BaseExcess, HCO3, WBC, Hct, Hgb, PTT, Fibrinogen, Platelets |
| Temperature    | Temp |

---

## 16-Week Timeline

| Weeks | Milestone |
|-------|-----------|
| 1     | Dataset feasibility audit · literature review |
| 2     | Proposal document (abstract, related work, gap statement) |
| 2–3   | EDA · missingness visualization |
| 3–4   | Preprocessing pipeline (masks, time-deltas, normalization, splits) |
| 4–5   | Organ-grouped feature construction · shock index |
| 5–6   | Baselines: XGBoost + plain Transformer |
| 6–7   | **MVP:** organ-grouped features → single Transformer |
| 7–8   | Add missingness + time-delta encoding (defensible fallback result) |
| 8–9   | Split into dual-branch architecture |
| 9–10  | Cross-attention fusion layer |
| 10–11 | Multi-task head (sepsis + organ-dysfunction auxiliary) |
| 11–12 | Full ablation suite (5-model table) |
| 12–13 | MC Dropout uncertainty · calibration (ECE) |
| 13    | Case-study interpretability · timing analysis |
| 13–14 | Results, discussion, figures |
| 14–15 | Full draft assembly |
| 15–16 | Polish · presentation prep |

---

## Baselines & Ablation

| Model | Organ Branch | Transformer | Missingness Encoding |
|-------|:---:|:---:|:---:|
| XGBoost | ❌ | ❌ | ❌ |
| Plain Transformer (naive imputation) | ❌ | ✅ | ❌ |
| Time-aware Transformer | ❌ | ✅ | ✅ |
| Organ Branch Only | ✅ | ❌ | ❌ |
| **Hybrid (Proposed)** | ✅ | ✅ | ✅ |

---

## Evaluation Metrics

- **Primary:** PhysioNet 2019 Utility Score
- **Secondary:** AUROC, AUPRC, Precision/Recall
- Timing analysis: hours-before-onset for true positives
- Calibration: ECE, reliability diagrams
- Qualitative: 3–5 patient case studies

---

## Target Venues

- Artificial Intelligence in Medicine
- Computers in Biology and Medicine
- Biomedical Signal Processing and Control
- Journal of Biomedical Informatics
- IEEE Journal of Biomedical and Health Informatics

---

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start — Week 1 Audit

```bash
python preprocessing/missingness_audit.py
```

This will scan all ~40k patients and produce:
- Per-variable missingness table (console + CSV)
- Per-organ-group missingness summary
- Class imbalance report
- Patient length statistics
- Figures saved to `figures/`
