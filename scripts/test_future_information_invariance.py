"""
scripts/test_future_information_invariance.py
-----------------------------------------------
Strict Future-Information Invariance Unit Test.

Verifies that for any time step t in a patient sequence:
  Prediction p(t) under X[0:t] + X[t+1:] (original future)
  equals
  Prediction p(t) under X[0:t] + X_modified[t+1:] (randomized future).

Requirement:
  abs(pred_A[t] - pred_B[t]) < 1e-5 for all t in [0, T-1].
"""

import sys
import torch
import numpy as np
from pathlib import Path

base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir))

from models.transformer.tact_model import TACTModel
from models.m5.m5_model import M5Model
from models.m5.temporal_experts import LocalTemporalExpert, GlobalTemporalExpert


def test_tact_model_causality():
    print("=== TEST 1: TACTModel Causal Invariance Test ===")
    torch.manual_seed(42)
    B, T, F = 2, 24, 102
    model = TACTModel(input_dim=F, d_model=64, num_layers=3)
    model.eval()

    # Original sequence
    x_orig = torch.randn(B, T, F)

    with torch.no_grad():
        out_orig = torch.sigmoid(model(x_orig))  # (B, T)

    # Test at each time step t from 0 to T-2
    for t in range(T - 1):
        # Create modified future sequence (randomize t+1 onwards)
        x_mod = x_orig.clone()
        x_mod[:, t + 1 :, :] = torch.randn(B, T - (t + 1), F) * 5.0 + 10.0

        with torch.no_grad():
            out_mod = torch.sigmoid(model(x_mod))

        # Check prediction at time step t
        diff = torch.abs(out_orig[:, t] - out_mod[:, t]).max().item()
        assert diff < 1e-5, f"CAUSALITY LEAKAGE AT t={t}! Max diff: {diff:.6e}"

    print("[PASS] TACTModel satisfies 100% causal future-information invariance!")


def test_m5_global_expert_causality():
    print("=== TEST 2: M5 GlobalTemporalExpert Causal Invariance Test ===")
    torch.manual_seed(42)
    B, T, D = 2, 24, 64
    expert = GlobalTemporalExpert(d_model=D, num_layers=2)
    expert.eval()

    x_orig = torch.randn(B, T, D)

    with torch.no_grad():
        out_orig = expert(x_orig)  # (B, T, D)

    for t in range(T - 1):
        x_mod = x_orig.clone()
        x_mod[:, t + 1 :, :] = torch.randn(B, T - (t + 1), D) * 10.0

        with torch.no_grad():
            out_mod = expert(x_mod)

        diff = torch.abs(out_orig[:, t, :] - out_mod[:, t, :]).max().item()
        assert diff < 1e-5, f"GLOBAL EXPERT LEAKAGE AT t={t}! Max diff: {diff:.6e}"

    print("[PASS] M5 GlobalTemporalExpert satisfies 100% causal future-information invariance!")


def test_m5_local_expert_causality():
    print("=== TEST 3: M5 LocalTemporalExpert (Causal Conv1D) Invariance Test ===")
    torch.manual_seed(42)
    B, T, D = 2, 24, 64
    expert = LocalTemporalExpert(in_dim=D, hidden_dim=D)
    expert.eval()

    x_orig = torch.randn(B, T, D)

    with torch.no_grad():
        out_orig = expert(x_orig)

    for t in range(T - 1):
        x_mod = x_orig.clone()
        x_mod[:, t + 1 :, :] = torch.randn(B, T - (t + 1), D) * 10.0

        with torch.no_grad():
            out_mod = expert(x_mod)

        diff = torch.abs(out_orig[:, t, :] - out_mod[:, t, :]).max().item()
        assert diff < 1e-5, f"LOCAL EXPERT LEAKAGE AT t={t}! Max diff: {diff:.6e}"

    print("[PASS] M5 LocalTemporalExpert satisfies 100% causal future-information invariance!")


def test_m5_model_causality():
    print("=== TEST 4: Full M5Model Causal Invariance Test ===")
    torch.manual_seed(42)
    B, T, F = 2, 24, 102
    model = M5Model(input_dim=F)
    model.eval()

    x_orig = torch.randn(B, T, F)

    with torch.no_grad():
        logits, _ = model(x_orig)
        out_orig = torch.sigmoid(logits)

    for t in range(T - 1):
        x_mod = x_orig.clone()
        x_mod[:, t + 1 :, :] = torch.randn(B, T - (t + 1), F) * 5.0

        with torch.no_grad():
            logits_mod, _ = model(x_mod)
            out_mod = torch.sigmoid(logits_mod)

        diff = torch.abs(out_orig[:, t] - out_mod[:, t]).max().item()
        assert diff < 1e-5, f"M5Model LEAKAGE AT t={t}! Max diff: {diff:.6e}"

    print("[PASS] Full M5Model satisfies 100% causal future-information invariance!")


if __name__ == "__main__":
    try:
        test_tact_model_causality()
        test_m5_global_expert_causality()
        test_m5_local_expert_causality()
        test_m5_model_causality()
        print("\n=======================================================")
        print("ALL STRICT CAUSAL TEMPORAL INVARIANCE TESTS PASSED 100%!")
        print("=======================================================")
    except AssertionError as e:
        print(f"\n[FAIL] Causality test failed: {e}")
        sys.exit(1)
