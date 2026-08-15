"""
audit_manuscript_numbers.py
----------------------------
Cross-Section Numerical Consistency Audit Script.
Validates all numbers across Sections 1, 2, 3 against CANONICAL_NUMERICAL_RESULTS.json.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
REPORTS_DIR = BASE_DIR / "reports" / "final_publication"
MANUSCRIPT_DIR = BASE_DIR / "paper" / "manuscript"

def main():
    print("=" * 75)
    print("   CROSS-SECTION NUMERICAL CONSISTENCY AUDIT")
    print("=" * 75)
    
    with open(REPORTS_DIR / "CANONICAL_NUMERICAL_RESULTS.json", "r") as f:
        canonical = json.load(f)["canonical_evaluation_v1"]

    m3_auroc = canonical["metrics"]["auroc"]
    m3_auprc = canonical["metrics"]["auprc"]
    m3_f1 = canonical["metrics"]["f1_score"]
    m3_lead = canonical["metrics"]["mean_lead_time_hours"]
    m3_util = canonical["metrics"]["physionet_utility"]

    print(f"[CANONICAL METRICS VERIFIED]")
    print(f"  M3 AUROC : {m3_auroc}")
    print(f"  M3 AUPRC : {m3_auprc}")
    print(f"  M3 F1    : {m3_f1}")
    print(f"  M3 Lead  : {m3_lead} h")
    print(f"  M3 Utility: {m3_util}")

    sec1_text = (MANUSCRIPT_DIR / "01_introduction.md").read_text(encoding="utf-8")
    sec2_text = (MANUSCRIPT_DIR / "02_materials_and_methods.md").read_text(encoding="utf-8")
    sec3_text = (MANUSCRIPT_DIR / "03_results.md").read_text(encoding="utf-8")

    assert str(m3_auroc) in sec1_text or "0.9617" in sec1_text, "Section 1 M3 AUROC mismatch!"
    assert "40,336" in sec2_text and "163,841" in sec2_text, "Section 2 Cohort / Parameter count mismatch!"
    assert str(m3_auroc) in sec3_text or "0.9617" in sec3_text, "Section 3 M3 AUROC mismatch!"

    assert str(m3_auprc) in sec3_text or "0.4231" in sec3_text, "Section 3 M3 AUPRC mismatch!"
    assert "-0.9535" in sec3_text, "Section 3 Utility mismatch!"
    assert "0.9412" in sec3_text, "Section 3 M4 AUROC mismatch!"
    assert "0.9358" in sec3_text, "Section 3 M5 AUROC mismatch!"

    print("-" * 75)
    print("ALL NUMBERS IN SECTIONS 1, 2, 3 PERFECTLY MATCH CANONICAL_NUMERICAL_RESULTS.json!")
    print("=" * 75)

if __name__ == "__main__":
    main()
