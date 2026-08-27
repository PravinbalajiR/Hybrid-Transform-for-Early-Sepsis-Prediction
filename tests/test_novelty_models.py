"""
tests/test_novelty_models.py
-----------------------------
Unit tests for PITACTModel:
  - Tensor shapes & multi-horizon heads
  - Strict causal invariance test on PITACTModel
"""

import sys
from pathlib import Path
base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir))

import unittest
import torch

from models.novelty.physio_transformer import PITACTModel


class TestPITACTModel(unittest.TestCase):

    def test_forward_shapes(self):
        B, T, F = 2, 24, 102
        x = torch.randn(B, T, F)
        model = PITACTModel(num_raw_features=34, d_model=64)
        out = model(x)

        self.assertIn("logits_6h", out)
        self.assertIn("logits_12h", out)
        self.assertIn("logits_24h", out)
        self.assertEqual(out["logits_6h"].shape, (B, T))
        self.assertEqual(out["logits_12h"].shape, (B, T))
        self.assertEqual(out["logits_24h"].shape, (B, T))

    def test_pitact_causality_invariance(self):
        torch.manual_seed(42)
        B, T, F = 2, 24, 102
        model = PITACTModel(num_raw_features=34, d_model=64)
        model.eval()

        x_orig = torch.randn(B, T, F)

        with torch.no_grad():
            out_orig = torch.sigmoid(model(x_orig)["logits_6h"])

        for t in range(T - 1):
            x_mod = x_orig.clone()
            x_mod[:, t + 1 :, :] = torch.randn(B, T - (t + 1), F) * 10.0

            with torch.no_grad():
                out_mod = torch.sigmoid(model(x_mod)["logits_6h"])

            diff = torch.abs(out_orig[:, t] - out_mod[:, t]).max().item()
            self.assertLess(diff, 1e-5, f"PITACT CAUSALITY LEAKAGE AT t={t}!")


if __name__ == "__main__":
    unittest.main()
