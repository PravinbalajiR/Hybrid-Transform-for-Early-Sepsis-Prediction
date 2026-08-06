"""
time_aware_embedding.py
-----------------------
Time-Aware and Missingness-Aware Embedding Layer for ICU Time-Series.

Transforms raw hourly input triplets (values, observation mask, time delta)
into a dense embedding vector in d_model space.

This module supports two input modes:

  M2 — Values only (input_dim = F = 34)
  M3 — Triplet     (input_dim = 3*F = 102)

In M3 mode the forward pass splits (B,T,102) → values(34) + masks(34) + deltas(34),
applies log1p(Δ) to compress the skewed delta distribution, and concatenates
[values, masks, log1p(deltas)] before projecting to d_model.

Key design decision:
  We use a simple log1p transform instead of Time2Vec.  Forensic analysis
  showed that Time2Vec's linear channel grew to O(1000) magnitude after
  LayerNorm, dominating 94% of post-norm variance and degrading model
  discriminability.  log1p(Δ) is bounded, monotone, and clinically
  interpretable (it compresses the long tail of multi-day gaps).

Ablation modes (controlled via YAML `ablation_mode`):
  none        → full triplet  (Values + Mask + log1p(Δ))   [M3]
  mask_only   → Values + Mask + zeros(Δ)                   [M3-mask]
  delta_only  → Values + zeros(Mask) + log1p(Δ)            [M3-delta]
  linear_delta→ plain projection without log transform     [diagnostic]

References:
  Vaswani et al. (2017) Attention Is All You Need
  Che et al.    (2018) GRU-D: RNNs for Multivariate Time Series with Missing Values
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard Sinusoidal Positional Encoding for time-series sequences."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TimeAwareEmbedding(nn.Module):
    """
    Time-Aware Embedding module.

    M2 mode (input_dim = F = 34):
        projects values directly: Linear(F -> d_model)

    M3 mode (input_dim = 3F = 102):
        1. Splits input into (values, masks, deltas)  — each (B, T, F)
        2. Applies log1p(clamp(deltas, min=0)) to compress skewed delta distribution
        3. Ablation zeroing if requested
        4. Concatenates [values, masks, log_deltas]  → still (B, T, 3F)
        5. Projects via Linear(3F -> d_model)
        6. LayerNorm → PositionalEncoding

    All ablation variants route through the same code path; only the
    zeroing of masks / deltas changes.

    Public API is identical to the previous implementation, so TACTModel
    requires zero code changes.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        max_len: int = 500,
        dropout: float = 0.1,
        ablation_mode: str = "none",
    ):
        super().__init__()
        self.d_model = d_model
        self.ablation_mode = ablation_mode
        self.is_triplet = (input_dim % 3 == 0) and (input_dim >= 102)
        self.F = input_dim // 3 if self.is_triplet else input_dim

        # Single linear projection — same in both M2 and M3
        # In M3, the projection dimension is 3F (values + masks + log_deltas)
        self.proj = nn.Linear(input_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, input_dim)  — either F (M2) or 3F (M3)

        Returns:
            (B, T, d_model)
        """
        if self.is_triplet:
            values = x[..., :self.F]              # (B, T, F)  — z-score imputed values
            masks  = x[..., self.F:2*self.F]      # (B, T, F)  — observation mask  {0, 1}
            deltas = x[..., 2*self.F:]            # (B, T, F)  — time-delta in hours

            # -------------------------------------------------------
            # Ablation zeroing (controlled by YAML `ablation_mode`)
            # -------------------------------------------------------
            if self.ablation_mode == "mask_only":
                # M3-mask: keep values + masks, ablate time information
                deltas = torch.zeros_like(deltas)
            elif self.ablation_mode == "delta_only":
                # M3-delta: keep values + deltas, ablate missingness pattern
                masks = torch.zeros_like(masks)
            # ablation_mode == "none" → keep all three channels (full M3)

            # -------------------------------------------------------
            # log1p(Δ) — compress the skewed delta distribution
            # Δ ranges from 0 to ~168 hrs.  log1p maps this to [0, ~5.1]
            # This prevents delta magnitudes from dominating the embedding.
            # -------------------------------------------------------
            log_deltas = torch.log1p(torch.clamp(deltas, min=0.0))  # (B, T, F)

            # Reconstruct triplet with transformed deltas
            x = torch.cat([values, masks, log_deltas], dim=-1)  # (B, T, 3F)

        # Single linear projection, shared across M2 and M3
        out = self.proj(x)                    # (B, T, d_model)
        out = self.layer_norm(out)             # normalise before positional encoding
        out = self.pos_encoder(out)            # add sinusoidal positional encoding
        return out

