"""
compile_full_manuscript.py
--------------------------
Compiles all manuscript sections (01-06), title, abstract, highlights, keywords,
and publication tables into a single unified manuscript document.
"""

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MANUSCRIPT_DIR = BASE_DIR / "paper" / "manuscript"
REPORTS_DIR = BASE_DIR / "reports" / "final_publication"

def main():
    print("Compiling full manuscript draft...")
    
    title = "# Time-Aware Representation of Irregular Physiological Data for Early Sepsis Prediction\n\n"
    
    highlights = (REPORTS_DIR / "HIGHLIGHTS.md").read_text(encoding="utf-8") + "\n\n---\n\n"
    
    abstract = """## Abstract

**Background:** Early identification of sepsis in the intensive care unit (ICU) is critical for initiating timely resuscitation. However, clinical time-series data present extreme missingness and irregular measurement intervals ($\Delta t$), which standard machine learning models flatten via static imputation.

**Objective:** This study investigates whether explicitly representing physiological values, observation missingness patterns ($\mathbf{m}$), and continuous temporal gaps ($\boldsymbol{\Delta t}$) within a Transformer architecture improves early sepsis prediction, and evaluates whether increasing architectural complexity yields superior performance.

**Methods:** We established a leak-free benchmark on $40,336$ ICU patients from the PhysioNet 2019 dataset (Train: $18,302$, Val: $2,034$, Test: $20,000$). We evaluated a baseline gradient boosted tree (M1), a plain Transformer (M2), a proposed Time-Aware Transformer (M3) incorporating continuous frequency temporal embeddings (Time2Vec) and missingness masks, four component ablation variants, and two exploratory multi-branch architectures (M4 Organ Hybrid and M5 Multi-Hybrid Network). Operating thresholds ($th=0.60$) were locked strictly on validation performance before single-pass test evaluation.

**Results:** M3 achieved the highest discrimination ($\text{AUROC} = 0.9617$, 95% CI: `[0.9495, 0.9727]`; $\text{AUPRC} = 0.4231$, 95% CI: `[0.3359, 0.5185]`), outperforming M1 ($\text{AUROC} = 0.8420$) and M2 ($\text{AUROC} = 0.9265$). Component ablations demonstrated that Time2Vec deltas extended mean lead time (+1.0 hour, reaching 5.2h in M3-Time+Delta and 5.7h in M3-Full), while missingness masks improved precision (+0.0449 PPV in M3-Full). M3 maintained superior calibration ($\text{ECE} = 0.0407$) and PhysioNet utility ($\text{Utility} = -0.9535$). Multi-branch MoE expert routing (M5) extended lead time to 12.0 hours but quadrupled false alarm rates ($5.80\%$ vs $1.83\%$ FPR/h) and degraded precision down to $11.58\%$.

**Conclusion:** Explicitly embedding temporal gaps and missingness patterns within a compact Transformer significantly enhances early sepsis alerting. Increasing architectural complexity via multi-branch MoE routing does not improve overall clinical utility.

**Keywords:** early sepsis prediction; intensive care; temporal modeling; missing data; deep learning; clinical prediction

---

"""

    sec1 = (MANUSCRIPT_DIR / "01_introduction.md").read_text(encoding="utf-8") + "\n\n---\n\n"
    sec2 = (MANUSCRIPT_DIR / "02_materials_and_methods.md").read_text(encoding="utf-8") + "\n\n---\n\n"
    sec3 = (MANUSCRIPT_DIR / "03_results.md").read_text(encoding="utf-8") + "\n\n---\n\n"
    sec4 = (MANUSCRIPT_DIR / "04_discussion.md").read_text(encoding="utf-8") + "\n\n---\n\n"
    sec5 = (MANUSCRIPT_DIR / "05_conclusions.md").read_text(encoding="utf-8") + "\n\n---\n\n"
    sec6 = (MANUSCRIPT_DIR / "06_references.md").read_text(encoding="utf-8")

    full_md = title + highlights + abstract + sec1 + sec2 + sec3 + sec4 + sec5 + sec6
    
    out_file = MANUSCRIPT_DIR / "FULL_MANUSCRIPT_DRAFT.md"
    out_file.write_text(full_md, encoding="utf-8")
    print(f"  -> Compiled full manuscript draft to: {out_file}")

if __name__ == "__main__":
    main()
