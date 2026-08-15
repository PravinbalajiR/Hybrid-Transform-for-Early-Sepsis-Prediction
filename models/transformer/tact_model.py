"""
transformer_encoder.py
----------------------
PyTorch Transformer Encoder for Early Sepsis Prediction.

Serves as the core temporal backbone for:
  - Model M2: Plain Transformer (naive mean imputation)
  - Model M3: Time-Aware Transformer (values + mask + time-delta triplet)

Architecture:
  1. TimeAwareEmbedding (projects input -> d_model + positional encoding)
  2. Multi-Layer Transformer Encoder (Self-Attention + FeedForward + LayerNorm)
  3. Per-Hour Prediction Head (Linear -> Sigmoid) -> outputs P(Sepsis within 6h)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from models.transformer.time_aware_embedding import TimeAwareEmbedding


class TACTModel(nn.Module):
    """
    Temporal Transformer Encoder for hourly sepsis risk scoring.
    """

    def __init__(
        self,
        input_dim: int = 34,        # 34 for plain, 34*3 = 102 for triplet time-aware
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_len: int = 500,
        ablation_mode: str = "none",
    ):
        super().__init__()
        self.d_model = d_model

        # Embedding layer
        self.embedding = TimeAwareEmbedding(
            input_dim=input_dim,
            d_model=d_model,
            max_len=max_len,
            dropout=dropout,
            ablation_mode=ablation_mode,
        )

        # Transformer Encoder Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,  # Input shape: (B, T, d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        # Output prediction head (per hour)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor,                          # (B, T, input_dim)
        padding_mask: Optional[torch.Tensor] = None, # (B, T) True for padded positions
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, input_dim)
            padding_mask: BoolTensor of shape (B, T) where True indicates padded positions

        Returns:
            logits: Output logits of shape (B, T)
        """
        # 1. Project and add positional encoding -> (B, T, d_model)
        h = self.embedding(x)

        # 2. Pass through Transformer Encoder
        # PyTorch src_key_padding_mask expects True at padded positions
        h_out = self.transformer_encoder(h, src_key_padding_mask=padding_mask)

        # 3. Compute per-hour logits -> (B, T)
        logits = self.fc_out(h_out).squeeze(-1)
        return logits


# Backward compatibility alias
SepsisTransformer = TACTModel


if __name__ == "__main__":
    B, T, F = 4, 48, 34
    x_plain = torch.randn(B, T, F)
    x_triplet = torch.randn(B, T, 3 * F)
    pad_mask = torch.zeros(B, T, dtype=torch.bool)
    pad_mask[:, 40:] = True  # pad last 8 hours

    model_plain = SepsisTransformer(input_dim=F)
    model_triplet = SepsisTransformer(input_dim=3 * F)

    out_p = model_plain(x_plain, padding_mask=pad_mask)
    out_t = model_triplet(x_triplet, padding_mask=pad_mask)

    print(f"Plain Transformer output logits shape  : {out_p.shape}")
    print(f"Triplet Transformer output logits shape: {out_t.shape}")
