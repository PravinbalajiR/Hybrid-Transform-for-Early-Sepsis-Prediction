"""
tests/test_shock_index.py
--------------------------
Unit tests for canonical Shock Index calculation:
  Shock Index = HR / (SBP + 1e-5)
"""

import torch
import unittest


def compute_shock_index(hr: torch.Tensor, sbp: torch.Tensor) -> torch.Tensor:
    """Canonical, safe Shock Index calculation."""
    return torch.clamp(hr / (torch.abs(sbp) + 1e-5), min=0.0, max=5.0)


class TestShockIndex(unittest.TestCase):

    def test_normal_case(self):
        hr = torch.tensor([90.0])
        sbp = torch.tensor([120.0])
        si = compute_shock_index(hr, sbp)
        self.assertAlmostEqual(si.item(), 0.75, places=4)

    def test_zero_sbp(self):
        hr = torch.tensor([90.0])
        sbp = torch.tensor([0.0])
        si = compute_shock_index(hr, sbp)
        self.assertFalse(torch.isnan(si).any())
        self.assertFalse(torch.isinf(si).any())
        self.assertEqual(si.item(), 5.0)

    def test_near_zero_sbp(self):
        hr = torch.tensor([90.0])
        sbp = torch.tensor([0.001])
        si = compute_shock_index(hr, sbp)
        self.assertFalse(torch.isnan(si).any())
        self.assertEqual(si.item(), 5.0)

    def test_high_instability(self):
        hr = torch.tensor([140.0])
        sbp = torch.tensor([80.0])
        si = compute_shock_index(hr, sbp)
        self.assertAlmostEqual(si.item(), 1.75, places=4)


if __name__ == "__main__":
    unittest.main()
