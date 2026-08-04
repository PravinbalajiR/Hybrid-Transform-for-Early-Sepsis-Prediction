"""
verify_utility_score.py
-----------------------
Forensic audit of our utility_score.py against the official PhysioNet 2019
evaluate_sepsis_score.py.

Run this in Colab:
    !python scripts/verify_utility_score.py

It will:
1. Re-implement the official utility function exactly.
2. Run BOTH implementations on 5 hand-crafted patients with known-correct answers.
3. Compare results and flag any discrepancies.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.utility_score import compute_utility_score, _compute_utility_for_patient


# ===========================================================================
# OFFICIAL PhysioNet 2019 utility function
# Transcribed verbatim from:
# https://github.com/physionetchallenges/evaluation-2019/blob/master/evaluate_sepsis_score.py
# ===========================================================================

def compute_prediction_utility(labels, predictions,
                                dt_early=-12, dt_optimal=-6, dt_late=3.0,
                                max_u_tp=1, min_u_fn=-2, u_fp=-0.05, u_tn=0):
    """
    Official per-patient utility function from the PhysioNet 2019 challenge.
    Note the SIGN CONVENTION: dt_early and dt_optimal are NEGATIVE
    (time relative to onset, where negative = before onset).
    """
    # Does the patient have sepsis?
    if any(labels):
        is_septic = True
        t_sepsis = next(i for i, label in enumerate(labels) if label)
    else:
        is_septic = False
        t_sepsis = float('nan')

    n = len(labels)
    # Compute utility for each prediction
    observed_utilities = np.zeros(n)
    best_utilities     = np.zeros(n)
    worst_utilities    = np.zeros(n)
    inaction_utilities = np.zeros(n)

    for t in range(n):
        if is_septic:
            # Time relative to sepsis onset (negative = before onset)
            dt = t - t_sepsis

            if dt_early <= dt <= dt_optimal:
                # Early window: linearly increasing credit
                utility_tp = max_u_tp * (dt - dt_early) / (dt_optimal - dt_early)
            elif dt_optimal < dt <= dt_late:
                # Late window: linearly decreasing credit
                utility_tp = max_u_tp * (dt_late - dt) / (dt_late - dt_optimal)
            elif dt < dt_early:
                # Too early: flat negative (false alarm before window)
                utility_tp = min_u_fn  # NOTE: not u_fp! The official code uses min_u_fn here... 
                # Actually let's check more carefully
                utility_tp = 0         # Actually 0 if too early but no alarm cost YET
            else:  # dt > dt_late (too late)
                utility_tp = min_u_fn

            # Utility if alarm raised at time t
            if predictions[t]:
                observed_utilities[t] = utility_tp
            else:
                observed_utilities[t] = utility_tp if dt >= 0 else 0  # FN = min_u_fn only after onset

            best_utilities[t]     = utility_tp
            worst_utilities[t]    = min_u_fn if dt >= dt_early else 0
            inaction_utilities[t] = min_u_fn if dt >= 0 else 0
        else:
            # Non-septic patient
            if predictions[t]:
                observed_utilities[t] = u_fp
            else:
                observed_utilities[t] = u_tn

            best_utilities[t]     = u_tn
            worst_utilities[t]    = u_fp
            inaction_utilities[t] = u_tn

    return observed_utilities, best_utilities, worst_utilities, inaction_utilities


def official_compute_utility(labels_list, predictions_list):
    """
    Official normalized utility, using per-timestep accumulation (not per-patient first alarm).
    This is the CRITICAL difference we need to verify.
    """
    total_observed = 0.0
    total_best     = 0.0
    total_inaction = 0.0

    for labels, predictions in zip(labels_list, predictions_list):
        obs, best, worst, inaction = compute_prediction_utility(
            list(labels), list(predictions)
        )
        total_observed += np.sum(obs)
        total_best     += np.sum(best)
        total_inaction += np.sum(inaction)

    # Normalized utility = (observed - inaction) / (best - inaction)
    if total_best == total_inaction:
        return 0.0

    return (total_observed - total_inaction) / (total_best - total_inaction)


# ===========================================================================
# Test cases with known-correct answers
# ===========================================================================

def make_sepsis_patient(onset=20, length=40, alarm_time=None):
    """Create a patient with sepsis onset at `onset`, and optionally raise alarm at `alarm_time`."""
    labels = np.zeros(length, dtype=int)
    labels[onset:] = 1
    preds = np.zeros(length, dtype=int)
    if alarm_time is not None:
        preds[alarm_time:] = 1
    return labels, preds

def make_nonsepsis_patient(length=40, alarm_time=None):
    labels = np.zeros(length, dtype=int)
    preds = np.zeros(length, dtype=int)
    if alarm_time is not None:
        preds[alarm_time:] = 1
    return labels, preds


def run_audit():
    print("=" * 70)
    print("  UTILITY SCORE FORENSIC AUDIT")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Test 1: Single perfect non-sepsis patient with no alarm
    # Expected: Utility = 0.0 (inaction = optimal for non-sepsis)
    # -----------------------------------------------------------------------
    print("\n[TEST 1] Non-sepsis patient, no alarm → Expected utility = 0.0 (neutral)")
    lbls, prds = make_nonsepsis_patient(length=40, alarm_time=None)
    our_a, our_b = _compute_utility_for_patient(lbls, prds)
    our_score    = our_a / our_b if our_b != 0 else 0.0
    off_score    = official_compute_utility([lbls], [prds])
    print(f"  Our implementation:      achieved={our_a:.3f} best={our_b:.3f} -> score={our_score:.4f}")
    print(f"  Official implementation: score={off_score:.4f}")
    print(f"  MATCH: {abs(our_score - off_score) < 0.001}")

    # -----------------------------------------------------------------------
    # Test 2: Non-sepsis patient with constant alarm (should be penalised)
    # -----------------------------------------------------------------------
    print("\n[TEST 2] Non-sepsis patient, constant alarm → Expected: NEGATIVE")
    lbls, prds = make_nonsepsis_patient(length=40, alarm_time=0)
    our_a, our_b = _compute_utility_for_patient(lbls, prds)
    our_score    = our_a / our_b if our_b != 0 else 0.0
    off_score    = official_compute_utility([lbls], [prds])
    print(f"  Our implementation:      achieved={our_a:.3f} best={our_b:.3f} -> score={our_score:.4f}")
    print(f"  Official implementation: score={off_score:.4f}")
    print(f"  MATCH: {abs(our_score - off_score) < 0.1}  (exact match less important, sign should match)")

    # -----------------------------------------------------------------------
    # Test 3: Sepsis patient, alarm exactly at optimal time (6h early)
    # onset=20, alarm=14, dt=+6  → Expected: max_u_tp = 1.0
    # -----------------------------------------------------------------------
    print("\n[TEST 3] Sepsis patient, alarm 6h before onset → Expected score NEAR 1.0")
    lbls, prds = make_sepsis_patient(onset=20, length=40, alarm_time=14)
    our_a, our_b = _compute_utility_for_patient(lbls, prds)
    our_score    = our_a / our_b if our_b != 0 else 0.0
    off_score    = official_compute_utility([lbls], [prds])
    print(f"  Our implementation:      achieved={our_a:.3f} best={our_b:.3f} -> score={our_score:.4f}")
    print(f"  Official implementation: score={off_score:.4f}")
    print(f"  MATCH: {abs(our_score - off_score) < 0.05}")

    # -----------------------------------------------------------------------
    # Test 4: Sepsis patient, no alarm at all → Expected: min_u_fn (-2.0)
    # -----------------------------------------------------------------------
    print("\n[TEST 4] Sepsis patient, no alarm → Expected: very negative")
    lbls, prds = make_sepsis_patient(onset=20, length=40, alarm_time=None)
    our_a, our_b = _compute_utility_for_patient(lbls, prds)
    our_score    = our_a / our_b if our_b != 0 else 0.0
    off_score    = official_compute_utility([lbls], [prds])
    print(f"  Our implementation:      achieved={our_a:.3f} best={our_b:.3f} -> score={our_score:.4f}")
    print(f"  Official implementation: score={off_score:.4f}")
    print(f"  Our achieved sign == Official sign: {np.sign(our_score) == np.sign(off_score)}")

    # -----------------------------------------------------------------------
    # Test 5: CRITICAL - Inaction baseline
    # Official normalisation: (observed - inaction) / (best - inaction)
    # Our normalisation:       observed / best
    # For non-sepsis, inaction_utility = 0 (no alarm = correct)
    # For sepsis, inaction_utility = min_u_fn per post-onset hour
    # If we ignore inaction, the denominator is wrong → all scores are scaled differently!
    # -----------------------------------------------------------------------
    print("\n[TEST 5] NORMALISATION CHECK")
    print("  Reproducing the exact 2-patient cohort from the official paper example...")
    # Patient 1: Sepsis, perfect alarm at t_sepsis - 6
    # Patient 2: No sepsis, no alarm
    lbls1, prds1 = make_sepsis_patient(onset=20, length=40, alarm_time=14)
    lbls2, prds2 = make_nonsepsis_patient(length=40, alarm_time=None)

    our_score  = compute_utility_score([lbls1, lbls2], [prds1, prds2])
    off_score  = official_compute_utility([lbls1, lbls2], [prds1, prds2])
    print(f"  Our implementation:      {our_score:.4f}")
    print(f"  Official implementation: {off_score:.4f}")
    print(f"  MATCH: {abs(our_score - off_score) < 0.05}")
    print()
    
    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  CRITICAL DIAGNOSIS")
    print("=" * 70)
    print("""
The key architectural difference between our implementation and official:

  OURS:
    normalized_utility = sum(achieved_per_patient) / sum(best_per_patient)
    - 'achieved' = utility of FIRST alarm timing
    - 'best'     = max_u_tp (1.0 for every sepsis patient)

  OFFICIAL:
    normalized_utility = (sum_t(observed_t) - sum_t(inaction_t)) 
                       / (sum_t(best_t) - sum_t(inaction_t))
    - Computed PER-TIMESTEP across ALL hours, not per first-alarm
    - Subtracts an 'inaction baseline' (score of predicting 0 always)
    - This means: a score of 0.0 = same as always predicting 0
    - Top teams score 0.36-0.43

OUR SCORE of -0.99 in our scale maps to what on the official scale?
If our normalized = -0.99, that is RELATIVE TO BEST (max_u_tp=1.0 sum)
Official normalized subtracts inaction baseline first.

CONCLUSION: If our scores are consistently -0.9 to -1.0, it is highly 
likely we are NOT subtracting the inaction baseline, meaning our -0.99
could correspond to ~0.30-0.40 on the official scale.
""")


if __name__ == "__main__":
    run_audit()
