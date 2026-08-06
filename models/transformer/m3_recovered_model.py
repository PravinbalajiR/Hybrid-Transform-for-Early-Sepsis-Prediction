"""
m3_recovered_model.py
---------------------
Exact Historical Model Architecture (M3 Recovered / Legacy TACT).
Recovered from checkpoint `best_time_aware_transformer_auroc0.973_epoch25.pt` (run_20260802_073034).

Features:
  - Direct Linear Embedding: nn.Linear(102, 64) mapping concatenated [Values, Masks, Deltas]
  - LayerNorm(64) + Sinusoidal PositionalEncoding
  - TransformerEncoder (3 layers, 4 heads, dim_feedforward=128, dropout=0.1)
  - FC Classifier Head (fc_out): Linear(64, 32) -> ReLU -> Dropout -> Linear(32, 1)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 64, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class LegacyTimeAwareEmbedding(nn.Module):
    def __init__(self, input_dim: int = 102, d_model: int = 64):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.pos_encoder = PositionalEncoding(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer_norm(self.proj(x) * math.sqrt(64))
        return self.pos_encoder(out)


class M3RecoveredModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 102,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = LegacyTimeAwareEmbedding(input_dim=input_dim, d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        emb = self.embedding(x)
        out = self.transformer_encoder(emb, src_key_padding_mask=padding_mask)
        return self.fc_out(out).squeeze(-1)


